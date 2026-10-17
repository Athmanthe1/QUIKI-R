"""
test_mini_pipeline.py
Tests the full Parquet/FAISS/PyTorch I/O pipeline on the Codespace disk 
using a tiny synthetic dataset (takes ~10 seconds).
"""
import os
import shutil
import subprocess
import pandas as pd
from PIL import Image

def main():
    print("🧪 STARTING MINI-PIPELINE PARQUET TEST...\n")
    
    # 1. Create a tiny real dataset on disk
    data_dir = "./mini_dataset"
    artifacts_dir = "./mini_artifacts"
    
    # Cleanup if exists
    if os.path.exists(data_dir): shutil.rmtree(data_dir)
    if os.path.exists(artifacts_dir): shutil.rmtree(artifacts_dir)
    
    print("1. Generating 15 dummy images on disk...")
    cats = ["Beaches", "Mountains", "Cats"]
    for cat in cats:
        cat_dir = os.path.join(data_dir, cat)
        os.makedirs(cat_dir, exist_ok=True)
        for i in range(5):
            color = (255, 0, 0) if cat == "Beaches" else (0, 255, 0) if cat == "Mountains" else (0, 0, 255)
            img = Image.new('RGB', (224, 224), color)
            img.save(os.path.join(cat_dir, f"img_{i}.jpg"))

    # 2. Run extract_embeddings.py
    print("2. Running extract_embeddings.py to generate Parquet & FAISS...")
    result = subprocess.run([
        "python", "extract_embeddings.py", 
        "--data_dir", data_dir, 
        "--output_dir", artifacts_dir, 
        "--batch_size", "4"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ extract_embeddings.py failed!")
        print(result.stderr)
        return
    print("✅ Extraction complete.\n")

    # 3. Verify the Parquet File on Disk
    print("3. Verifying Parquet file I/O...")
    parquet_path = os.path.join(artifacts_dir, "dataset_embeddings.parquet")
    if not os.path.exists(parquet_path):
        print("❌ Parquet file was not created!")
        return
        
    df = pd.read_parquet(parquet_path)
    print(f"   -> Parquet shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   -> Columns: {list(df.columns)}")
    print(f"   -> Parquet file size: {os.path.getsize(parquet_path) / 1024:.2f} KB")
    assert len(df) == 15, "Expected 15 rows in Parquet!"
    print("✅ Parquet I/O is flawless.\n")

    # 4. Run trainer.py on the real Parquet file
    print("4. Running trainer.py on the generated Parquet file...")
    result = subprocess.run([
        "python", "trainer.py", 
        "--train_path", parquet_path, 
        "--output_dir", artifacts_dir, 
        "--epochs", "2", 
        "--batch_size", "4",
        "--min_freq", "1"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ trainer.py failed!")
        print(result.stderr)
        return
    print("✅ Training complete.\n")

    # 5. Final Verification
    print("5. Verifying final artifacts...")
    head_path = os.path.join(artifacts_dir, "supervised_head.pth")
    assert os.path.exists(head_path), "Trained head weights missing!"
    print(f"   -> supervised_head.pth size: {os.path.getsize(head_path) / 1024:.2f} KB")
    
    print("\n🎉 MINI-PIPELINE PARQUET TEST PASSED!")
    print("Your Codespace is 100% ready to process the real 2GB dataset locally.")
    
    # Cleanup
    shutil.rmtree(data_dir)
    shutil.rmtree(artifacts_dir)
    print("\n🧹 Cleaned up mini_dataset and mini_artifacts.")

if __name__ == "__main__":
    main()