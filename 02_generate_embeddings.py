import numpy as np
import pandas as pd
import umap
from tqdm.auto import tqdm
import os
import sys
import time
from openai import OpenAI
from openai import RateLimitError, APIError
from dotenv import load_dotenv

# Register tqdm with pandas
tqdm.pandas()

load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    print("FATAL ERROR: OPENAI_API_KEY environment variable not set. Please ensure it is defined in your .env file.")
    sys.exit(1)

CLIENT = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"
# The default dimension for text-embedding-3-small is 1536, but we can use 256 for lower cost/faster UMAP
EMBEDDING_DIM = 256 
print(f"Using OpenAI Embedding Model: {EMBEDDING_MODEL} (Dimension: {EMBEDDING_DIM})")

def get_openai_embeddings_batch(code_list, batch_size=500):
    all_embeddings = []
    
    # Process the data in smaller batches to avoid the 300k token limit
    for i in tqdm(range(0, len(code_list), batch_size), desc="OpenAI API Batches"):
        batch_codes = code_list[i:i + batch_size]
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = CLIENT.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch_codes,
                    dimensions=EMBEDDING_DIM # Request smaller dimension embeddings
                )
                batch_embeddings = [d.embedding for d in response.data]
                all_embeddings.extend(batch_embeddings)
                break 

            except RateLimitError:
                wait_time = 2 ** attempt
                print(f"\nRate limit hit. Waiting {wait_time}s before retrying batch {i//batch_size}.")
                time.sleep(wait_time)
                
            except APIError as e:
                if 'max 300000 tokens per request' in str(e):
                    # This is the token limit error. The batch is too big.
                    print(f"\nFATAL API ERROR (Token Limit) on batch {i//batch_size}. Batch is too large. Reduce batch_size further.")
                else:
                    print(f"\nGeneral API Error on batch {i//batch_size}. Error: {e}")
                
                # Insert zero vectors as standard Python LISTS on error
                zero_vector = [0.0] * EMBEDDING_DIM
                zero_vectors_list = [zero_vector] * len(batch_codes)
                all_embeddings.extend(zero_vectors_list)
                break 
                
            except Exception as e:
                print(f"\nUnknown Error on batch {i//batch_size}. Error: {e}")
                
                # Insert zero vectors as standard Python LISTS on error
                zero_vector = [0.0] * EMBEDDING_DIM
                zero_vectors_list = [zero_vector] * len(batch_codes)
                all_embeddings.extend(zero_vectors_list)
                break

        else:
            print(f"\nMax retries exceeded for batch {i//batch_size}. Proceeding with zero vectors.")
            zero_vector = [0.0] * EMBEDDING_DIM
            zero_vectors_list = [zero_vector] * len(batch_codes)
            all_embeddings.extend(zero_vectors_list)
    return np.array(all_embeddings)


def create_stratified_sample(df, sample_fraction=1/16):
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
    
    try:
        df = pd.read_csv("results/structural_features.csv")
    except FileNotFoundError:
        print("\nERROR: Could not find 'results/structural_features.csv'. Please run 01_extract_features.py first.")
        sys.exit(1)

    # Sample 1/16th of the data for UMAP visualization
    print(f"\nTotal records: {len(df)}")
    df_sample = create_stratified_sample(df, sample_fraction=1/16.0) 
    df_sample = df_sample.dropna(subset=['code'])
    print(f"Sampled records (approx. 1/16th): {len(df_sample)}")

    df_sample['id'] = df_sample.index 

    print("\nStarting Code Embedding Generation (OpenAI API)")
    
    all_embeddings = get_openai_embeddings_batch(df_sample['code'].tolist(), batch_size=500)
    
    df_sample['embedding'] = list(all_embeddings)

    # Prepare Data for UMAP
    X_raw = np.stack(df_sample['embedding'].values)
    
    # Filter out zero vectors resulting from failed API calls
    non_zero_mask = ~np.all(X_raw == 0, axis=1) 
    X = X_raw[non_zero_mask] 
    
    if X.shape[0] == 0:
        print("\nFATAL ERROR: No valid embeddings were generated. Check your API key and network connection.")
        sys.exit(1)

    # Run UMAP only on non-zero vectors
    print(f"\nApplying UMAP for 2D Projection ({X.shape[0]} valid embeddings)")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    embedding_2d = reducer.fit_transform(X)

    # Add UMAP results back to the original filtered DataFrame
    df_final = df_sample[non_zero_mask].copy()
    df_final['UMAP 1'] = embedding_2d[:, 0]
    df_final['UMAP 2'] = embedding_2d[:, 1]

    output_cols = ['id', 'author_type', 'model', 'UMAP 1', 'UMAP 2']
    output_filename = "results/sample_embeddings.csv"
    df_final[output_cols].to_csv(output_filename, index=False)

    print(f"\nData processing complete. Sample embedding features saved to {output_filename} ({len(df_final)} records).")