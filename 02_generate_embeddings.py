# Save this as 02_generate_embeddings_sample.py (CPU Parallelized)

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import umap
from tqdm.auto import tqdm
import os
import warnings
import psutil
import sys
from joblib import Parallel, delayed # <--- NEW IMPORT

# Suppress PyTorch/Seaborn warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Register tqdm with pandas
tqdm.pandas()

# --- Optimization ---
# Set PyTorch/MKL thread count explicitly to maximize CPU usage
NUM_THREADS = psutil.cpu_count(logical=False) 
if NUM_THREADS is None:
    NUM_THREADS = psutil.cpu_count(logical=True) 
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS) 
os.environ['MKL_NUM_THREADS'] = str(NUM_THREADS) 
torch.set_num_threads(NUM_THREADS)

# IMPORTANT: Set the number of jobs for joblib to use (10 cores)
N_JOBS = NUM_THREADS
print(f"Set PyTorch/MKL thread count to: {NUM_THREADS} cores.")
print(f"Embedding generation will use {N_JOBS} parallel workers.")
# --- End Optimization ---


# Embedding Model Setup (Model and tokenizer must be loaded within each parallel process)
MODEL_NAME = "microsoft/codebert-base" 
print(f"Loading embedding model: {MODEL_NAME}...")

# WARNING: We must ensure the model is loaded on CPU and initialized *once* per process.
# We will define a function that does the loading, and joblib will handle the rest.

# Function to initialize model/tokenizer (will be run inside each joblib worker)
def init_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    # Ensure it's on CPU and evaluation mode
    model.to(torch.device("cpu"))
    model.eval()
    return tokenizer, model

# Single-sample embedding function (now accepts model/tokenizer)
def get_code_embedding(code, tokenizer, model):
    """Generates a dense vector representation of the code."""
    try:
        inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Device is explicitly CPU here
        device = torch.device("cpu") 
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding
        
    except Exception as e:
        # Note: Added error logging for debugging
        print(f"\nEmbedding generation failed for a code snippet. Error: {e}")
        return np.zeros(model.config.hidden_size)


def create_stratified_sample(df, sample_fraction=1/16):
    """
    Creates a sample that is evenly stratified across all 'model' types.
    """
    if 'model' not in df.columns:
        print("Warning: 'model' column not found for stratification. Using simple random sample.")
        return df.sample(frac=sample_fraction, random_state=42)
    
    df_sample = (
        df.groupby('model', group_keys=False)
        .apply(lambda x: x.sample(frac=sample_fraction, random_state=42))
    ).reset_index(drop=True)
    
    return df_sample


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    # Load data from the first script
    try:
        df = pd.read_csv("results/structural_features.csv")
    except FileNotFoundError:
        print("\nERROR: Could not find 'results/structural_features.csv'.")
        print("Please run 01_extract_features.py first.")
        sys.exit(1)

    # --- Sample the data (1/16th of the data) ---
    print(f"\nTotal records: {len(df)}")
    df_sample = create_stratified_sample(df, sample_fraction=1/16.0) 
    print(f"Sampled records (approx. 1/16th): {len(df_sample)}")

    df_sample['id'] = df_sample.index 
    df_sample = df_sample.dropna(subset=['code'])


    # --- Embedding Generation with Joblib Parallelization ---
    print(f"\n--- 3. Starting Code Embedding Generation (CodeBERT) with {N_JOBS} parallel workers ---")
    
    # Initialize the model once here (Note: joblib will often re-initialize in workers)
    # The actual parallel execution:
    # 1. Use joblib's Parallel to run tasks concurrently.
    # 2. Use 'delayed' to specify the function and arguments for each task.
    # 3. The `__init__` function is a helper to ensure the model/tokenizer is present in each worker.
    
    # Use Parallel with a helper function to initialize model/tokenizer per joblib worker
    all_embeddings = Parallel(n_jobs=N_JOBS)(
        delayed(lambda code: get_code_embedding(code, *init_model()))(code)
        for code in tqdm(df_sample['code'], desc="Embedding Generation")
    )

    df_sample['embedding'] = all_embeddings


    # Prepare Data for UMAP (Rest of the script is unchanged)
    X_raw = np.stack(df_sample['embedding'].values)
    non_zero_mask = ~np.all(X_raw == 0, axis=1) 
    X = X_raw[non_zero_mask] 
    
    if X.shape[0] == 0:
        print("\nFATAL ERROR: All sampled embeddings resulted in zero vectors. Cannot run UMAP.")
        sys.exit(1)

    # Run UMAP only on non-zero vectors
    print("\n--- 4. Applying UMAP for 2D Projection ---")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    embedding_2d = reducer.fit_transform(X)

    # Add UMAP results back to the original filtered DataFrame
    df_final = df_sample[non_zero_mask].copy()
    df_final['UMAP 1'] = embedding_2d[:, 0]
    df_final['UMAP 2'] = embedding_2d[:, 1]
    
    # --- SAVE ONLY EMBEDDING RESULTS + ID ---
    output_cols = ['id', 'author_type', 'model', 'UMAP 1', 'UMAP 2']
    output_filename = "results/sample_embeddings.csv"
    df_final[output_cols].to_csv(output_filename, index=False)

    print(f"\nData processing complete. Sample embedding features saved to {output_filename} ({len(df_final)} records).")