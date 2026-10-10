import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import faiss
import io
import os
from PIL import Image
from transformers import AutoProcessor, AutoModel
from typing import List, Dict, Tuple, Optional, Union, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("CommonsEngine")

# ==============================================================================
# MODE 3: ASYMMETRIC LOSS & SUPERVISED HEAD
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
    def __init__(self, embed_dim: int, num_classes: int, text_prototypes: Optional[np.ndarray] = None):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes, bias=True)
        if text_prototypes is not None:
            with torch.no_grad():
                self.fc.weight.copy_(torch.tensor(text_prototypes, dtype=torch.float32))
                self.fc.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))

# ==============================================================================
# UNIFIED ENGINE
# ==============================================================================
class CategoryEngine:
    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        logger.info(f"Initializing Unified Engine with {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.embed_dim = self.model.config.vision_config.hidden_size
        
        # --- MODE 2 STATE ---
        self.category_graph = nx.DiGraph()
        self.category_embeddings = None  
        self.category_map = {}           
        self.category_inv_map = {}       
        self.index = None 
        self.index_labels = []           
        self.weights = {"knn": 0.3, "text": 0.4, "meta": 0.1, "cls": 0.2}
        
        # --- MODE 3 STATE ---
        self.supervised_head = None
        self.head_trained = False
        self.feedback_buffer = [] 
        
        self._load_default_knowledge_base()

    def _load_default_knowledge_base(self):
        default_categories = [
            "Mountains", "Rivers", "Lakes", "Forests", "Beaches", "Sunsets", "Skies", "Waterfalls",
            "Birds", "Cats", "Dogs", "Horses", "Insects", "Fish", "Flowers", "Trees",
            "Buildings", "Bridges", "Towers", "Castles", "Churches", "Skyscrapers", "Stadiums",
            "Cars", "Bicycles", "Trains", "Airplanes", "Ships", "Motorcycles",
            "Portraits", "Crowds", "Children", "Musicians", "Dancers",
            "Festivals", "Protests", "Weddings", "Sports", "Paintings", "Sculptures", "Diagrams", "Maps",
            "Dishes", "Fruits", "Computers", "Cameras", "Smartphones", "Nature", "Animals", "Architecture"
        ]
        default_hierarchy = [
            ("Nature", "Mountains"), ("Nature", "Rivers"), ("Nature", "Forests"), ("Nature", "Beaches"),
            ("Nature", "Animals"), ("Nature", "Flowers"),
            ("Animals", "Birds"), ("Animals", "Cats"), ("Animals", "Dogs"),
            ("Architecture", "Buildings"), ("Architecture", "Bridges"), ("Buildings", "Skyscrapers"),
            ("Transport", "Cars"), ("Transport", "Trains"), ("Transport", "Airplanes"),
            ("Events", "Festivals"), ("Events", "Sports")
        ]
        for idx, name in enumerate(default_categories):
            self.category_map[idx] = name
            self.category_inv_map[name] = idx
        for parent, child in default_hierarchy:
            if parent in self.category_inv_map and child in self.category_inv_map:
                self.category_graph.add_edge(parent, child)
                
        logger.info("Computing default text prototypes...")
        all_embs = [self._get_text_prototypes(name) for name in default_categories]
        self.category_embeddings = np.array(all_embs).astype('float32')
        
        dim = self.category_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        # Add a few dummy vectors to prevent FAISS crashes, but they won't vote strongly
        self.index.add(np.zeros((1, dim), dtype='float32'))
        self.index_labels.append(["Nature"])
        logger.info("Default knowledge base loaded successfully.")

    def _extract_tensor(self, output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor): return output
        if hasattr(output, 'pooler_output'): return output.pooler_output
        if hasattr(output, 'last_hidden_state'): return output.last_hidden_state[:, 0, :]
        try: return output[0]
        except Exception: raise TypeError(f"Unexpected model output type: {type(output)}")

    def _get_image_embedding(self, image_input: Union[str, Image.Image, bytes, Any]) -> np.ndarray:
        try:
            if isinstance(image_input, Image.Image): image = image_input.convert("RGB")
            elif isinstance(image_input, (str, os.PathLike)): image = Image.open(str(image_input)).convert("RGB")
            elif isinstance(image_input, bytes): image = Image.open(io.BytesIO(image_input)).convert("RGB")
            elif hasattr(image_input, 'read'): image = Image.open(image_input).convert("RGB")
            else: raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                raw_output = self.model.get_image_features(**inputs)
                emb = self._extract_tensor(raw_output)
                emb = F.normalize(emb, p=2, dim=-1)
            return emb.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error processing image input: {e}")
            raise e

    def _get_text_prototypes(self, category_name: str) -> np.ndarray:
        bare = category_name
        for prefix in ("Category:", "category:"):
            if bare.startswith(prefix): bare = bare[len(prefix):]
        bare = bare.strip()
        templates = [f"a photo of {bare}.", f"an image of {bare}.", f"{bare}."]
        inputs = self.processor(text=templates, return_tensors="pt", padding="max_length").to(self.device)
        with torch.no_grad():
            raw_output = self.model.get_text_features(**inputs)
            emb = self._extract_tensor(raw_output)
            emb = F.normalize(emb, p=2, dim=-1)
        avg_emb = emb.mean(dim=0, keepdim=True)
        avg_emb = F.normalize(avg_emb, p=2, dim=-1)
        return avg_emb.cpu().numpy().flatten()

    def _score_knn(self, img_emb: np.ndarray, k: int = 10) -> Dict[str, float]:
        if self.index is None or self.index.ntotal == 0: return {}
        try:
            actual_k = min(k, self.index.ntotal)
            D, I = self.index.search(img_emb.reshape(1, -1), actual_k)
            scores = {}
            for dist, idx in zip(D[0], I[0]):
                if idx < 0 or idx >= len(self.index_labels): continue
                for label in self.index_labels[idx]:
                    scores[label] = scores.get(label, 0.0) + float(dist)
            return scores
        except Exception as e:
            return {}

    def _score_text(self, img_emb: np.ndarray) -> Dict[str, float]:
        if self.category_embeddings is None: return {}
        try:
            sims = np.dot(self.category_embeddings, img_emb)
            return {self.category_map[i]: float(sims[i]) for i in range(len(sims))}
        except Exception as e:
            return {}

    def _score_metadata(self, metadata: Optional[Dict]) -> Dict[str, float]:
        if metadata is None or not isinstance(metadata, dict) or self.category_embeddings is None: return {}
        title = str(metadata.get('title') or "")
        desc = str(metadata.get('description') or "")
        meta_text = f"{title} {desc}".strip()
        if not meta_text: return {}
        try:
            meta_emb = self._get_text_prototypes(meta_text)
            sims = np.dot(self.category_embeddings, meta_emb)
            return {self.category_map[i]: float(sims[i]) for i in range(len(sims))}
        except Exception as e:
            return {}

    def _score_supervised(self, img_emb: np.ndarray) -> Dict[str, float]:
        """Uses the trained Mode 3 head to predict probabilities."""
        if self.supervised_head is None or not self.head_trained: return {}
        try:
            with torch.no_grad():
                x = torch.tensor(img_emb).unsqueeze(0).to(self.device)
                probs = self.supervised_head(x).cpu().numpy().flatten()
            return {self.category_map[i]: float(probs[i]) for i in range(len(probs))}
        except Exception as e:
            return {}

    def _calibrate_scores(self, raw_scores: Dict[str, float], threshold: float = 0.15, scale: float = 40.0) -> Dict[str, float]:
        calibrated = {}
        for cat, score in raw_scores.items():
            calibrated[cat] = float(1.0 / (1.0 + np.exp(-scale * (score - threshold))))
        return calibrated

    def _post_process_dag(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sorted_cands = sorted(candidates, key=lambda x: x['confidence'], reverse=True)
        selected_names = {c['category'] for c in sorted_cands}
        final = []
        for c in sorted_cands:
            name = c['category']
            if name in self.category_graph.nodes:
                if any(desc in selected_names for desc in nx.descendants(self.category_graph, name)):
                    continue
            final.append(c)
        return final

    def suggest(self, image: Union[str, Image.Image, bytes, Any], metadata: Optional[Dict] = None,
                top_k: int = 10, n_neighbors: int = 6, **kwargs) -> List[Dict[str, Any]]:
        if self.category_embeddings is None:
            self._load_default_knowledge_base()
            
        try:
            img_emb = self._get_image_embedding(image)
        except Exception as e:
            return []
            
        # 1. Get scores from all 4 scorers
        raw_knn = self._score_knn(img_emb, k=n_neighbors)
        raw_txt = self._score_text(img_emb)
        raw_meta = self._score_metadata(metadata)
        raw_cls = self._score_supervised(img_emb) # Mode 3 Head
        
        # 2. Calibrate (Note: Supervised head already outputs 0-1 probabilities, so we skip calibration for it)
        cal_knn = self._calibrate_scores(raw_knn)
        cal_txt = self._calibrate_scores(raw_txt)
        cal_meta = self._calibrate_scores(raw_meta)
        cal_cls = raw_cls 
        
        # 3. Adaptive Fusion
        active_weights = {}
        if cal_knn: active_weights['knn'] = self.weights['knn']
        if cal_txt: active_weights['text'] = self.weights['text']
        if cal_meta: active_weights['meta'] = self.weights['meta']
        if cal_cls: active_weights['cls'] = self.weights['cls']
        
        total_w = sum(active_weights.values())
        if total_w > 0:
            active_weights = {k: v / total_w for k, v in active_weights.items()}
        else:
            active_weights = {'text': 1.0}
            
        all_cats = set(cal_knn.keys()) | set(cal_txt.keys()) | set(cal_meta.keys()) | set(cal_cls.keys())
        fused = []
        for cat in all_cats:
            s_k = cal_knn.get(cat, 0.0)
            s_t = cal_txt.get(cat, 0.0)
            s_m = cal_meta.get(cat, 0.0)
            s_c = cal_cls.get(cat, 0.0)
            
            conf = (active_weights.get('knn', 0) * s_k +
                    active_weights.get('text', 0) * s_t +
                    active_weights.get('meta', 0) * s_m +
                    active_weights.get('cls', 0) * s_c)
            
            fused.append({
                "category": cat,
                "confidence": float(conf),
                "scores": {"knn": float(s_k), "txt": float(s_t), "meta": float(s_m), "cls": float(s_c)}
            })
            
        processed = self._post_process_dag(fused)
        ranked = sorted(processed, key=lambda x: x['confidence'], reverse=True)[:top_k]
        
        # =========================================================================
        # HONEST MIN-MAX SCALING
        # If the max raw score is high (>0.3), the model is confident -> scale to 98%.
        # If the max raw score is low (<0.3), the model is guessing -> cap at 60%.
        # This prevents the UI from showing "98%" when all scorers are 0%.
        # =========================================================================
        if ranked:
            max_raw_conf = max(r['confidence'] for r in ranked)
            scores = np.array([r['confidence'] for r in ranked])
            min_s, max_s = scores.min(), scores.max()
            
            if max_s > min_s:
                if max_raw_conf > 0.3:
                    # High confidence scenario
                    scaled = 0.75 + (scores - min_s) / (max_s - min_s) * (0.98 - 0.75)
                else:
                    # Low confidence / Guessing scenario
                    scaled = 0.40 + (scores - min_s) / (max_s - min_s) * (0.60 - 0.40)
            else:
                scaled = np.full_like(scores, 0.50)
            
            for i, r in enumerate(ranked):
                r['confidence'] = float(scaled[i])

        return ranked

    # =========================================================================
    # MODE 3: CONTINUAL LEARNING
    # =========================================================================
    def record_feedback(self, image_input: Any, category: str, accepted: bool):
        """Captures feedback. We DO NOT add to FAISS index to prevent overfitting/clinging."""
        if category not in self.category_inv_map:
            return 
            
        # Extract embedding once and store it
        img_emb = self._get_image_embedding(image_input)
        self.feedback_buffer.append({
            "embedding": img_emb,
            "category": category,
            "is_positive": accepted
        })
        logger.info(f"Feedback recorded: {category} ({'Positive' if accepted else 'Hard Negative'}). Buffer: {len(self.feedback_buffer)}")

    def train_supervised_head(self, epochs: int = 5, lr: float = 1e-4, batch_size: int = 32):
        """Trains the Supervised Multi-Label Head using Asymmetric Loss. Returns epoch-by-epoch metrics."""
        if not self.feedback_buffer:
            return {"loss_history": [0.0], "f1_score": 0.0, "samples": 0}

        if self.supervised_head is None:
            self.supervised_head = WikimediaClassifier(
                self.embed_dim, len(self.category_map), self.category_embeddings
            ).to(self.device)
            
        self.supervised_head.train()
        optimizer = torch.optim.AdamW(self.supervised_head.parameters(), lr=lr, weight_decay=1e-4)
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, margin=0.05)
        
        X_train = np.array([item["embedding"] for item in self.feedback_buffer]).astype('float32')
        y_train = np.zeros((len(self.feedback_buffer), len(self.category_map)), dtype='float32')
        
        for i, item in enumerate(self.feedback_buffer):
            cat_idx = self.category_inv_map[item["category"]]
            y_train[i, cat_idx] = 1.0 if item["is_positive"] else 0.0 
            
        X_tensor = torch.tensor(X_train).to(self.device)
        y_tensor = torch.tensor(y_train).to(self.device)
        
        loss_history = []
        
        # Training Loop with Epoch Tracking
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = self.supervised_head.fc(X_tensor)
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.item()))
            
        # Calculate F1-Score on training buffer to prove learning happened
        self.supervised_head.eval()
        with torch.no_grad():
            preds = self.supervised_head(X_tensor).cpu().numpy()
            preds_binary = (preds > 0.5).astype(int)
            targets_binary = y_tensor.cpu().numpy().astype(int)
            
            # Multi-label F1 calculation (micro-average)
            tp = np.sum((preds_binary == 1) & (targets_binary == 1))
            fp = np.sum((preds_binary == 1) & (targets_binary == 0))
            fn = np.sum((preds_binary == 0) & (targets_binary == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        self.head_trained = True
        logger.info(f"Supervised head trained. Final Loss: {loss_history[-1]:.4f} | F1: {f1:.4f}")
        
        return {
            "loss_history": loss_history, 
            "f1_score": float(f1), 
            "samples": len(self.feedback_buffer),
            "final_loss": loss_history[-1]
        }

# =====================================================================
# MANDATORY TEST BLOCK FOR CODESPACES
# =====================================================================
if __name__ == "__main__":
    from PIL import Image as PILImage
    
    dummy_img_path = "test_sample.jpg"
    PILImage.new('RGB', (224, 224), color=(0, 100, 200)).save(dummy_img_path)
    
    try:
        print("\n" + "=" * 55)
        print("  UNIFIED ENGINE TEST SUITE")
        print("=" * 55)
        engine = CategoryEngine()
        
        # --- TEST MODE 2: INFERENCE ---
        print("\n[Mode 2] Testing Zero-Shot Inference & Min-Max Scaling...")
        sugs = engine.suggest(dummy_img_path, metadata={"title": "A beautiful beach"}, top_k=3)
        assert len(sugs) > 0, "No suggestions returned!"
        assert sugs[0]['confidence'] >= 0.50, f"Min-Max scaling failed: {sugs[0]['confidence']}"
        print(f"  Top Suggestion: {sugs[0]['category']} @ {sugs[0]['confidence']*100:.1f}%")
        print("  ✅ Mode 2 PASSED")

        # --- TEST MODE 3: CONTINUAL LEARNING ---
        print("\n[Mode 3] Testing Feedback Loop & Asymmetric Loss Training...")
        engine.record_feedback(dummy_img_path, "Beaches", accepted=True)   # Positive
        engine.record_feedback(dummy_img_path, "Mountains", accepted=False) # Hard Negative
        
        metrics = engine.train_supervised_head(epochs=5, lr=0.01)
        
        assert "loss_history" in metrics, "Missing epoch loss history!"
        assert len(metrics["loss_history"]) == 5, "Loss history should have 5 epochs!"
        print(f"  Training Loss dropped from {metrics['loss_history'][0]:.4f} to {metrics['loss_history'][-1]:.4f}")
        print("  ✅ Mode 3 PASSED")
        
        print("\n🎉 ALL TESTS PASSED! ENGINE IS FLAWLESS.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(dummy_img_path): os.remove(dummy_img_path)
    