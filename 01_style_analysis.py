import numpy as np
from datasets import load_dataset
from radon.complexity import cc_visit
import scipy.stats as stats
import ast
import warnings 
from transformers import AutoTokenizer, AutoModel
import umap
import torch
import pandas as pd
from tqdm.auto import tqdm
import os

# Suppress warnings, dataset has unicode \
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module='seaborn')


# Load and structure data
def load_and_structure_data():
    dataset = load_dataset("OSS-forge/HumanVsAICode", split="train")
    raw_df = dataset.to_pandas()
    rename_map = {"human_code": "human", "chatgpt_code": "chatgpt", "dsc_code": "deepseek", "qwen_code": "qwen"}
    
    df_long = raw_df.rename(columns=rename_map).melt(
        value_vars=["human", "chatgpt", "deepseek", "qwen"],
        var_name="model",
        value_name="code"
    ).dropna(subset=["code"])
    
    df_long["author_type"] = np.where(df_long["model"] == "human", "human", "ai")
    return df_long

#Extract code features
def extract_features(code):
    try:
        lines = code.split("\n")
        non_empty_lines = [l for l in lines if l.strip()]
        loc = len(lines)
        def safe_div(n, d): return n / d if d > 0 else 0
        
        avg_line_len = np.mean([len(l) for l in non_empty_lines]) if non_empty_lines else 0
        comments = [l for l in lines if l.strip().startswith("#")]
        comment_density = safe_div(len(comments), loc)
        indent_depths = [len(l) - len(l.lstrip()) for l in non_empty_lines]
        avg_indent = np.mean(indent_depths) if indent_depths else 0

        func_count, avg_var_len = 0, 0
        try:
            tree = ast.parse(code)
            func_count = sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
            var_names = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)]
            avg_var_len = np.mean([len(n) for n in var_names]) if var_names else 0
        except SyntaxError:
            func_count = sum(1 for l in non_empty_lines if l.strip().startswith("def "))
            
        avg_complexity, total_complexity = 0, 0
        try:
            blocks = cc_visit(code)
            complexities = [b.complexity for b in blocks]
            avg_complexity = np.mean(complexities) if complexities else 0
            total_complexity = sum(complexities)
        except Exception:
            pass

        return pd.Series({
            "loc": loc,
            "avg_line_len": avg_line_len,
            "comment_density": comment_density,
            "indent_depth": avg_indent,
            "function_count": func_count,
            "var_name_len": avg_var_len,
            "avg_cyclomatic": avg_complexity,
            "total_cyclomatic": total_complexity
        })

    except Exception:
        return pd.Series({k: 0 for k in ["loc", "avg_line_len", "comment_density", "indent_depth", "function_count", "var_name_len", "avg_cyclomatic", "total_cyclomatic"]})

# Embedding Model Setup
MODEL_NAME = "microsoft/codebert-base" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def get_code_embedding(code, model=model, tokenizer=tokenizer):
    try:
        inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model.to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding
    except Exception:
        return np.zeros(model.config.hidden_size)

tqdm.pandas()

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    print("Loading and Structuring Data...")
    df = load_and_structure_data()
    
    # --- Feature Extraction with Progress Bar ---
    print("\nStarting Feature Extraction (Structural and Complexity Metrics)...")
    # Change .apply() to .progress_apply()
    features_df = df["code"].progress_apply(extract_features)
    df = pd.concat([df, features_df], axis=1)

    # Normalization
    df["complexity_per_100_loc"] = (df["total_cyclomatic"] / df["loc"]) * 100
    df["funcs_per_100_loc"] = (df["function_count"] / df["loc"]) * 100
    
    # --- Embedding Generation with Progress Bar ---
    print("\nStarting Code Embedding Generation (CodeBERT)...")
    # Change .apply() to .progress_apply()
    df['embedding'] = df['code'].progress_apply(lambda x: get_code_embedding(x))

    # Prepare Data for UMAP
    X_raw = np.stack(df['embedding'].values)
    non_zero_mask = ~np.all(X_raw == 0, axis=1) 
    X = X_raw[non_zero_mask] 
    
    # Run UMAP only on non-zero vectors
    print("\nApplying UMAP for 2D Projection...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    embedding_2d = reducer.fit_transform(X)

    # Add UMAP results back to the original filtered DataFrame
    df_final = df[non_zero_mask].copy()
    df_final['UMAP 1'] = embedding_2d[:, 0]
    df_final['UMAP 2'] = embedding_2d[:, 1]
    
    # Save the data
    output_cols = ['author_type', 'UMAP 1', 'UMAP 2', 'loc', 'avg_line_len', 'comment_density', 'indent_depth', 'function_count', 'var_name_len', 'avg_cyclomatic', 'total_cyclomatic', 'complexity_per_100_loc', 'funcs_per_100_loc']
    df_final[output_cols].to_csv("results/analysis_data.csv", index=False)

    print(f"\nData processing complete. Results saved to analysis_data.csv ({len(df_final)} records).")