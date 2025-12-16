# Save this as 03_plot_umap.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
import os
import sys
import numpy as np 

# Silences the Seaborn FutureWarning for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module='seaborn')

# Set style
sns.set_theme(style="whitegrid")

# Create output directory
OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving plots to directory: {OUTPUT_DIR}")


# --- UMAP Plotting Function (FIXED FOR SCALE, TICKS, AND SIZE) ---
def plot_umap(data, hue_col='author_type', filename="umap_clustering_authortype.png", title_suffix="Author Type"):
    """
    Generates a UMAP scatter plot, configurable by 'author_type' or 'model', 
    with dynamic axis scaling and explicit 0.5 tick increments, saved at high resolution.
    """
    if data.empty:
        print(f"\nSkipping UMAP plot ({hue_col}): Sample data is empty.")
        return
        
    # --- CHANGE 1: INCREASED FIGURE SIZE FOR LARGER PLOT ---
    plt.figure(figsize=(14, 12)) 
    
    # Define distinct color palettes and order for the two plots
    if hue_col == 'author_type':
        palette = {'human': 'darkorange', 'ai': 'dodgerblue'}
        hue_order = ['human', 'ai'] 
    else: # hue_col == 'model'
        palette = {'human': '#FF9900', 'chatgpt': '#007ACC', 'deepseek': '#4CAF50', 'qwen': '#F44336'} 
        hue_order = ['human', 'chatgpt', 'deepseek', 'qwen'] 

    sns.scatterplot(
        x='UMAP 1', 
        y='UMAP 2', 
        hue=hue_col, 
        data=data, 
        palette=palette,
        hue_order=hue_order, 
        s=20, 
        alpha=0.6,
        linewidth=0 
    )

    # --- AXIS LIMITS AND TICK CONTROL (as previously fixed) ---
    x_min, x_max = data['UMAP 1'].min(), data['UMAP 1'].max()
    y_min, y_max = data['UMAP 2'].min(), data['UMAP 2'].max()
    
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    x_lim_min, x_lim_max = x_min - x_padding, x_max + x_padding
    y_lim_min, y_lim_max = y_min - y_padding, y_max + y_padding
    
    plt.xlim(x_lim_min, x_lim_max)
    plt.ylim(y_lim_min, y_lim_max)
    
    # Set Ticks at 0.5 increments
    x_ticks = np.arange(round(x_lim_min - 0.5), round(x_lim_max + 0.5), 0.5)
    y_ticks = np.arange(round(y_lim_min - 0.5), round(y_lim_max + 0.5), 0.5)
    
    x_ticks = x_ticks[(x_ticks >= x_lim_min) & (x_ticks <= x_lim_max)]
    y_ticks = y_ticks[(y_ticks >= y_lim_min) & (y_ticks <= y_lim_max)]
    
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    # -----------------------------------------------------------

    plt.title(f'Code Style Clustering (UMAP) - Colored by {title_suffix} (Sampled Data)', fontweight='bold')
    plt.legend(title=title_suffix)
    plt.tight_layout()
    # --- CHANGE 2: INCREASED DPI FOR HIGHER RESOLUTION OUTPUT ---
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=600)
    plt.close()
    print(f"  - Saved {filename}") 


if __name__ == "__main__":
    
    # --- 1. Load Data ---
    try:
        df_umap = pd.read_csv("results/sample_embeddings.csv")
        
        if not all(col in df_umap.columns for col in ['UMAP 1', 'UMAP 2', 'author_type', 'model']):
            print("ERROR: 'sample_embeddings.csv' is missing required columns.")
            sys.exit(1)
        
        print(f"Data prepared for UMAP Plot (sample set): {len(df_umap)} records.")

        print("\nModel Counts in UMAP Data:")
        print(df_umap['model'].value_counts())
        
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find required file: {e.filename}")
        print("Please ensure 02_generate_embeddings_sample.py has run successfully.")
        sys.exit(1)


    # --- 2. Generate UMAP Plots ---

    # UMAP Plot (Author Type - Human vs AI) 
    print("\n--- Generating Embedding Plot (UMAP by Author Type) ---")
    plot_umap(df_umap, hue_col='author_type', filename="umap_clustering_authortype.png", title_suffix="Author Type (Human vs AI)")


    # UMAP Plot (Model - Human vs 3 AIs)
    print("\n--- Generating Embedding Plot (UMAP by Specific Model) ---")
    plot_umap(df_umap, hue_col='model', filename="umap_clustering_model.png", title_suffix="Specific Model (Human vs 3 AIs)")

    print("\nVisualization complete. UMAP charts saved as individual PNG files.")