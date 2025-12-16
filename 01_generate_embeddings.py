# Save this as 02_generate_embeddings.py

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import umap
from tqdm.auto import tqdm
import os
import warnings
import psutil

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

def get_code_embedding(code, model=model, tokenizer=tokenizer):
    """Generates a dense vector representation of the code."""
    try:
        inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model.to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use Mean Pooling of the last hidden state
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding
    except Exception:
        return np.zeros(model.config.hidden_size)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    # Load data from the first script
    try:
        df = pd.read_csv("results/structural_features.csv.csv")
    except FileNotFoundError:
        print("\nERROR: Could not find 'results/structural_features.csv.csv'.")
        print("Please run 01_extract_features.py first.")
        exit()

    # --- Embedding Generation with Progress Bar ---
    print("\n--- 3. Starting Code Embedding Generation (CodeBERT) ---")
    df['embedding'] = df['code'].progress_apply(lambda x: get_code_embedding(x))

    # Prepare Data for UMAP
    X_raw = np.stack(df['embedding'].values)
    non_zero_mask = ~np.all(X_raw == 0, axis=1) 
    X = X_raw[non_zero_mask] 
    
    # Run UMAP only on non-zero vectors
    print("\n--- 4. Applying UMAP for 2D Projection ---")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    embedding_2d = reducer.fit_transform(X)

    # Add UMAP results back to the original filtered DataFrame
    df_final = df[non_zero_mask].copy()
    df_final['UMAP 1'] = embedding_2d[:, 0]
    df_final['UMAP 2'] = embedding_2d[:, 1]
    
    # Drop the code and embedding columns before saving the final results
    df_final = df_final.drop(columns=['code', 'embedding'])
    
    # Save the data
    output_cols = ['author_type', 'UMAP 1', 'UMAP 2', 'loc', 'avg_line_len', 'comment_density', 'indent_depth', 'function_count', 'var_name_len', 'avg_cyclomatic', 'total_cyclomatic', 'complexity_per_100_loc', 'funcs_per_100_loc']
    df_final[output_cols].to_csv("results/analysis_data.csv", index=False)

    print(f"\nData processing complete. Final analysis data saved to results/analysis_data.csv ({len(df_final)} records).")