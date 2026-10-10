"""
trainer.py - Large-scale offline training for the Supervised Multi-Label Head.
Usage: python trainer.py --train_path data/train.parquet --output_dir artifacts/

REQUIRES A GPU CLUSTER. Do NOT run on GitHub Codespaces CPU.
"""
import os
import json
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModel
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("OfflineTrainer")

# ==============================================================================
# 1. COMPONENTS (Mirrored from engine.py for standalone execution)
# ==============================================================================
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, margin=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.margin = margin
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pos_weight = (1 - probs) ** self.gamma_pos
        pos_loss = -targets * pos_weight * torch.log(probs + self.eps)
        probs_neg = torch.clamp(probs - self.margin, min=0.0)
        neg_weight = probs_neg ** self.gamma_neg
        neg_loss = -(1 - targets) * neg_weight * torch.log(1 - probs + self.eps)
        return (pos_loss + neg_loss).sum(dim=-1).mean()

class WikimediaClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, text_prototypes: np.ndarray = None):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes, bias=True)
        if text_prototypes is not None:
            with torch.no_grad():
                self.fc.weight.copy_(torch.tensor(text_prototypes, dtype=torch.float32))
                self.fc.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))

# ==============================================================================
# 2. DATASET (Handles chunking & broken image paths safely)
# ==============================================================================
class WikimediaDataset(Dataset):
    def __init__(self, parquet_path, processor, category_map, image_col="image_path", cat_col="categories"):
        self.df = pd.read_parquet(parquet_path)
        self.processor = processor
        self.category_map = category_map
        self.image_col = image_col
        self.cat_col = cat_col
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row[self.image_col]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (0,0,0)) # Fallback for broken paths
            
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        
        num_classes = len(self.category_map)
        target = torch.zeros(num_classes, dtype=torch.float32)
        
        cats = row[self.cat_col]
        if isinstance(cats, str):
            try: cats = json.loads(cats)
            except: cats = [cats]
            
        if isinstance(cats, (list, tuple, set)):
            for c in cats:
                if c in self.category_map:
                    target[self.category_map[c]] = 1.0
                    
        return pixel_values, target

# ==============================================================================
# 3. MAIN TRAINING LOOP
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Offline training for Wikimedia Supervised Head")
    parser.add_argument("--train_path", type=str, required=True, help="Path to train.parquet")
    parser.add_argument("--output_dir", type=str, default="artifacts", help="Where to save weights")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("WARNING: Running on CPU. This will be extremely slow. Use a GPU cluster for real training.")
        
    logger.info(f"Training on device: {device}")
    
    # 1. Load Frozen Encoder
    model_name = "google/siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
    embed_dim = encoder.config.vision_config.hidden_size
    
    # 2. Scan Categories to build the massive label space
    logger.info("Scanning training data to build label space...")
    train_df = pd.read_parquet(args.train_path)
    all_cats = set()
    for cats in train_df['categories']:
        if isinstance(cats, list): all_cats.update(cats)
        elif isinstance(cats, str):
            try: all_cats.update(eval(cats))
            except: pass
    categories = sorted(list(all_cats))
    category_map = {c: i for i, c in enumerate(categories)}
    num_classes = len(categories)
    logger.info(f"Label space size: {num_classes}")
    
    # 3. Text Prototypes for Weight Initialization
    logger.info("Computing text prototypes for weight initialization...")
    text_prototypes = []
    for i in range(0, len(categories), 64):
        batch = categories[i:i+64]
        templates = [f"a photo of {c}." for c in batch]
        inputs = processor(text=templates, return_tensors="pt", padding="max_length").to(device)
        with torch.no_grad():
            emb = encoder.get_text_features(**inputs)
            emb = F.normalize(emb, p=2, dim=-1)
        text_prototypes.append(emb.cpu().numpy())
    text_prototypes = np.vstack(text_prototypes).astype('float32')
    
    # 4. Initialize Head
    head = WikimediaClassifier(embed_dim, num_classes, text_prototypes).to(device)
    
    # 5. DataLoader
    train_dataset = WikimediaDataset(args.train_path, processor, category_map)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, margin=0.05)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    
    # 6. Training Loop
    head.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, (pixel_values, targets) in enumerate(train_loader):
            pixel_values = pixel_values.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            with torch.no_grad():
                img_embs = encoder.get_image_features(pixel_values=pixel_values)
                img_embs = F.normalize(img_embs, p=2, dim=-1)
                
            logits = head.fc(img_embs)
            loss = criterion(logits, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if step % 50 == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs} | Step {step} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
        
    # 7. Save Artifacts
    torch.save(head.state_dict(), os.path.join(args.output_dir, "supervised_head.pth"))
    with open(os.path.join(args.output_dir, "category_map.json"), "w") as f:
        json.dump(category_map, f)
    logger.info(f"Training complete. Artifacts saved to {args.output_dir}/")

if __name__ == "__main__":
    main()