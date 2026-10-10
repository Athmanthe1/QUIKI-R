"""
extract_embeddings.py
Memory-safe, chunked pipeline to extract SigLIP embeddings for a large local dataset.
Outputs: dataset_embeddings.parquet, faiss_index.index, faiss_labels.json, category_map.json
"""
import os
import gc
import json
import argparse
import logging
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import faiss
from PIL import Image
from transformers import AutoProcessor, AutoModel
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("EmbeddingExtractor")

# Configuration
MODEL_NAME = "google/siglip-base-patch16-224"
IMAGE_SIZE = 224
ALLOWED_EXTS = {'.jpg', '.jpeg', '.png'}

def extract_and_save(data_dir: str, output_dir: str, batch_size: int = 16):
    """
    Scans data_dir for images, extracts embeddings in chunks, and saves artifacts.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading frozen SigLIP encoder on {device}...")
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    embed_dim = model.config.vision_config.hidden_size
    
    # 1. Scan Dataset
    logger.info(f"Scanning {data_dir} for images...")
    image_paths = []
    categories_list = []
    
    # Assumes structure: data_dir/CategoryName/image.jpg
    # Adjust this loop if your dataset is structured differently (e.g., a CSV file).
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if Path(file).suffix.lower() in ALLOWED_EXTS:
                img_path = os.path.join(root, file)
                category = os.path.basename(root) 
                image_paths.append(img_path)
                categories_list.append(category)

    if not image_paths:
        raise FileNotFoundError(f"No images found in {data_dir}")
        
    logger.info(f"Found {len(image_paths)} images. Starting chunked extraction...")

    # 2. Chunked Extraction Loop
    all_embeddings = []
    all_paths = []
    all_cats = []
    unique_categories = set()
    
    # Initialize FAISS
    index = faiss.IndexFlatIP(embed_dim)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_cats = categories_list[i:i+batch_size]
        
        valid_imgs = []
        
        # Load and preprocess batch
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE)) 
                valid_imgs.append(img)
            except Exception:
                continue # Skip broken images safely

        if not valid_imgs:
            continue

        # Encode
        inputs = processor(images=valid_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            # Handle different HF output structures safely
            emb = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0]
            emb = F.normalize(emb, p=2, dim=-1).cpu().numpy().astype('float32')

        # Store and Index
        all_embeddings.extend(emb)
        all_paths.extend(batch_paths[:len(valid_imgs)]) # Match valid count
        all_cats.extend(batch_cats[:len(valid_imgs)])
        unique_categories.update(batch_cats[:len(valid_imgs)])
        
        index.add(emb)

        # MEMORY CLEANUP (Crucial for Codespaces)
        del inputs, outputs, emb, valid_imgs
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        if i % (batch_size * 10) == 0:
            logger.info(f"Processed {i}/{len(image_paths)} images...")

    # 3. Save Artifacts
    logger.info("Saving Parquet, FAISS index, and label mappings...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Parquet
    df = pd.DataFrame({
        "image_path": all_paths,
        "category": all_cats,
        "embedding": [emb.tolist() for emb in all_embeddings]
    })
    parquet_path = os.path.join(output_dir, "dataset_embeddings.parquet")
    df.to_parquet(parquet_path, index=False)
    
    # Save FAISS Index
    faiss_path = os.path.join(output_dir, "faiss_index.index")
    faiss.write_index(index, faiss_path)
    
    # Save FAISS Labels (Row index -> List of categories)
    # Note: In a real multi-label dataset, this would be a list of lists. 
    # Here we wrap single categories in a list to match engine.py expectations.
    faiss_labels = [[cat] for cat in all_cats]
    labels_path = os.path.join(output_dir, "faiss_labels.json")
    with open(labels_path, 'w') as f:
        json.dump(faiss_labels, f)
        
    # Save Category Map (Int ID -> Category Name)
    sorted_cats = sorted(list(unique_categories))
    category_map = {i: cat for i, cat in enumerate(sorted_cats)}
    map_path = os.path.join(output_dir, "category_map.json")
    with open(map_path, 'w') as f:
        json.dump(category_map, f)
        
    logger.info(f"✅ Done! Processed {len(df)} images. Artifacts saved to {output_dir}")
    return parquet_path, faiss_path, labels_path, map_path


def run_test():
    """Generates dummy images and verifies the extraction pipeline."""
    import tempfile
    import shutil
    
    logger.info("🧪 STARTING EMBEDDING EXTRACTION TEST...")
    
    # 1. Create dummy dataset
    temp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(temp_dir, "dummy_data")
    output_dir = os.path.join(temp_dir, "dummy_output")
    
    cats = ["Category:Beaches", "Category:Mountains", "Category:Cats"]
    for cat in cats:
        cat_dir = os.path.join(data_dir, cat)
        os.makedirs(cat_dir, exist_ok=True)
        # Create 5 dummy images per category
        for i in range(5):
            color = (255, 0, 0) if cat == "Category:Beaches" else (0, 255, 0) if cat == "Category:Mountains" else (0, 0, 255)
            img = Image.new('RGB', (224, 224), color)
            img.save(os.path.join(cat_dir, f"img_{i}.jpg"))
            
    logger.info(f"Created dummy dataset at {data_dir}")
    
    try:
        # 2. Run extraction
        parquet_path, faiss_path, labels_path, map_path = extract_and_save(data_dir, output_dir, batch_size=4)
        
        # 3. Verify outputs
        assert os.path.exists(parquet_path), "Parquet file missing!"
        assert os.path.exists(faiss_path), "FAISS index missing!"
        assert os.path.exists(labels_path), "Labels JSON missing!"
        assert os.path.exists(map_path), "Category map missing!"
        
        df = pd.read_parquet(parquet_path)
        assert len(df) == 15, f"Expected 15 rows, got {len(df)}"
        
        index = faiss.read_index(faiss_path)
        assert index.ntotal == 15, f"Expected 15 vectors in FAISS, got {index.ntotal}"
        
        with open(map_path, 'r') as f:
            cat_map = json.load(f)
        assert len(cat_map) == 3, f"Expected 3 categories, got {len(cat_map)}"
        
        logger.info("✅ EMBEDDING EXTRACTION TEST PASSED!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        logger.info("Cleaned up dummy files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SigLIP embeddings from a local dataset.")
    parser.add_argument("--data_dir", type=str, default="./wikimedia_dataset", help="Path to image dataset")
    parser.add_argument("--output_dir", type=str, default="./artifacts", help="Where to save artifacts")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for extraction (keep low for RAM safety)")
    parser.add_argument("--test", action="store_true", help="Run the dummy data test block")
    
    args = parser.parse_args()
    
    if args.test:
        run_test()
    else:
        extract_and_save(args.data_dir, args.output_dir, args.batch_size)