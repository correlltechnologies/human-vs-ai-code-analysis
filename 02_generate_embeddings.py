# Save this as 02_generate_embeddings_sample.py

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
print(f"Set PyTorch/MKL thread count to: {NUM_THREADS} cores.")
# --- End Optimization ---


# Embedding Model Setup
MODEL_NAME = "microsoft/codebert-base" 
print(f"Loading embedding model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# Updated get_code_embedding function

def get_code_embedding(code, model=model, tokenizer=tokenizer):
    """Generates a dense vector representation of the code."""
    try:
        inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # --- FIX: ALWAYS USE CPU FOR STABILITY ---
        # We will not check for CUDA, as the error is likely happening during device transfer.
        device = torch.device("cpu") 
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model.to(device) # Ensure the model is on CPU
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use Mean Pooling of the last hidden state
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding
        
    except Exception as e:
        print(f"\nEmbedding generation failed for a code snippet. Error: {e}")
        # Return zeros for any embedding failure
        return np.zeros(model.config.hidden_size)

def create_stratified_sample(df, sample_fraction=1/16):
    """
    Creates a sample that is evenly stratified across all 'model' types 
    (human, chatgpt, deepseek, qwen).
    """
    if 'model' not in df.columns:
        # Fall back to simple random sampling if 'model' column is missing
        print("Warning: 'model' column not found for stratification. Using simple random sample.")
        return df.sample(frac=sample_fraction, random_state=42)
    
    # Stratified sampling logic based on the 'model' column
    # We use sample(frac) on the grouped data to maintain the proportions
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
    # The sample_fraction is now 1/16.0 as requested
    df_sample = create_stratified_sample(df, sample_fraction=1/16.0) 
    print(f"Sampled records (approx. 1/16th): {len(df_sample)}")

    # Add a temporary ID for merging back later. The ID needs to be consistent with the 
    # structural file's index if you rely on the index for merging in the visualization script.
    # Since we loaded the whole file, the original index is lost. We must use the original index 
    # values if they were saved, or create a unique identifier if they were not.
    
    # Since the sample is a subset, we will use the index values of the original df 
    # which represent the 'id' saved in the structural_features.csv (implicit index)
    
    # Re-aligning the IDs based on the original index of the loaded dataframe:
    df_sample['id'] = df_sample.index 
    df_sample = df_sample.dropna(subset=['code'])


    # --- Embedding Generation with Progress Bar ---
    print("\n--- 3. Starting Code Embedding Generation (CodeBERT) on Sample ---")
    df_sample['embedding'] = df_sample['code'].progress_apply(lambda x: get_code_embedding(x))

    # Prepare Data for UMAP
    X_raw = np.stack(df_sample['embedding'].values)
    non_zero_mask = ~np.all(X_raw == 0, axis=1) 
    X = X_raw[non_zero_mask] 
    
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