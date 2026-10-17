"""
generate_mock_artifacts.py
Generates tiny, synthetic artifacts to test the local app.py integration 
WITHOUT downloading or processing a 2GB dataset.
"""
import os
import json
import torch
import numpy as np
import faiss

# 1. Setup
ARTIFACTS_DIR = "./artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# SigLIP base patch16-224 embedding dimension is 768
EMBED_DIM = 768 
NUM_FAKE_IMAGES = 10
NUM_FAKE_CATS = 3

# 2. Generate Fake Category Map
fake_cats = ["FakeCategory:Beaches", "FakeCategory:Mountains", "FakeCategory:Cats"]
category_map = {i: cat for i, cat in enumerate(fake_cats)}
with open(os.path.join(ARTIFACTS_DIR, "category_map.json"), "w") as f:
    json.dump(category_map, f)

# 3. Generate Fake FAISS Index & Labels
index = faiss.IndexFlatIP(EMBED_DIM)
fake_labels = []
for _ in range(NUM_FAKE_IMAGES):
    # Random normalized vector
    vec = np.random.randn(1, EMBED_DIM).astype('float32')
    vec /= np.linalg.norm(vec)
    index.add(vec)
    fake_labels.append([np.random.choice(fake_cats)]) # Random label

faiss.write_index(index, os.path.join(ARTIFACTS_DIR, "faiss_index.index"))
with open(os.path.join(ARTIFACTS_DIR, "faiss_labels.json"), "w") as f:
    json.dump(fake_labels, f)

# 4. Generate Fake Supervised Head Weights
# Matches the exact architecture of WikimediaClassifier in engine.py
head = torch.nn.Linear(EMBED_DIM, NUM_FAKE_CATS, bias=True)
torch.save(head.state_dict(), os.path.join(ARTIFACTS_DIR, "supervised_head.pth"))

print(f"✅ Mock artifacts generated in '{ARTIFACTS_DIR}/'!")
print("Run 'streamlit run app.py' to test the local production bridge.")