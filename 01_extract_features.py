# Save this as 01_extract_features.py

import numpy as np
import pandas as pd
from datasets import load_dataset
from radon.complexity import cc_visit
import ast
from tqdm.auto import tqdm
import os
import warnings
import psutil # For core count

# Register tqdm with pandas
tqdm.pandas()

# --- Optimization ---
# Set PyTorch/MKL thread count explicitly to maximize CPU usage
NUM_THREADS = psutil.cpu_count(logical=False) 
if NUM_THREADS is None:
    NUM_THREADS = psutil.cpu_count(logical=True) 
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS) 
os.environ['MKL_NUM_THREADS'] = str(NUM_THREADS) 
print(f"Set OMP/MKL thread count to: {NUM_THREADS} cores.")
# --- End Optimization ---

# Suppress warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)


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

# Extract code features
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
            "code": code, # Keep code for the next script
            "author_type": None, # Will be set below
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
        # Note: Return the code column in the failure case as well
        return pd.Series({k: (code if k == 'code' else 0) for k in ["code", "author_type", "loc", "avg_line_len", "comment_density", "indent_depth", "function_count", "var_name_len", "avg_cyclomatic", "total_cyclomatic"]})


# REVISED if __name__ == "__main__": block for 01_extract_features.py

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    print("\n--- 1. Loading and Structuring Data ---")
    df = load_and_structure_data() # df now contains 'author_type', 'model', and 'code'
    
    # --- Feature Extraction with Progress Bar ---
    print("\n--- 2. Starting Feature Extraction (Structural and Complexity Metrics) ---")
    
    # Apply features. features_df contains the extracted metrics.
    features_df = df["code"].progress_apply(extract_features) 
    
    # We no longer need to drop 'code' from df, but we need to select 
    # the new columns from features_df and combine them with the original df.
    
    # Identify the new metric columns created by extract_features (excluding 'code' and 'author_type' if present)
    new_metric_cols = [col for col in features_df.columns if col not in ['code', 'author_type']]
    
    # Concatenate the new metric columns to the original df
    df = pd.concat([df.reset_index(drop=True), features_df[new_metric_cols].reset_index(drop=True)], axis=1)

    # Normalization
    df["complexity_per_100_loc"] = (df["total_cyclomatic"] / df["loc"]) * 100
    df["funcs_per_100_loc"] = (df["function_count"] / df["loc"]) * 100
    
    # Save the data
    output_cols = ['author_type', 'code', 'loc', 'avg_line_len', 'comment_density', 'indent_depth', 'function_count', 'var_name_len', 'avg_cyclomatic', 'total_cyclomatic', 'complexity_per_100_loc', 'funcs_per_100_loc']
    
    # This line will now work because 'code' has been retained in df
    df[output_cols].to_csv("results/structural_features.csv", index=False)

    print(f"\nData processing complete. Structural features saved to results/structural_features.csv ({len(df)} records).")