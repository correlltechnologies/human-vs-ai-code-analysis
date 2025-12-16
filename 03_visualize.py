# Save this as 03_visualize.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
import os
import sys
import scipy.stats as stats # New import for statistical testing

# Silences the Seaborn FutureWarning for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module='seaborn')

# --- 1. Load and Merge Data (Same as before, relies on structural_features.csv containing 'model') ---

try:
    # 1. Load Structural Features (Full Dataset)
    df_struct = pd.read_csv("results/structural_features.csv")
    df_struct['id'] = df_struct.index 
    
    # 2. Load Sample Embedding Features (Only Sample)
    df_embed_sample = pd.read_csv("results/sample_embeddings.csv")
    
    # Check if the sample file exists, if not, create a placeholder for UMAP
    if 'UMAP 1' not in df_embed_sample.columns:
         print("\nWARNING: Sample embedding data is incomplete. UMAP plot will fail.")
    
    print(f"Loaded structural features (full set): {len(df_struct)} records.")
    print(f"Loaded embedding features (sample set): {len(df_embed_sample)} records.")

    # Merge the structural features with the embedding sample features on the generated ID
    df_final = pd.merge(
        df_struct, 
        df_embed_sample[['id', 'UMAP 1', 'UMAP 2']], 
        on='id', 
        how='left' 
    )
    df_final = df_final.drop(columns=['id'])

    # df_boxplot_umap: Full structural data for box plots, filtered embedding data for UMAP
    df_boxplot = df_final 
    df_umap = df_final.dropna(subset=['UMAP 1', 'UMAP 2'])

    print(f"Data prepared for Box Plots (full set): {len(df_boxplot)} records.")
    print(f"Data prepared for UMAP Plot (sample set): {len(df_umap)} records.")
    
except FileNotFoundError as e:
    print(f"\nERROR: Could not find required file: {e.filename}")
    print("Please ensure all previous scripts have run successfully.")
    sys.exit(1)

# Set style
sns.set_theme(style="whitegrid")

# Create output directory
OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving plots to directory: {OUTPUT_DIR}")


# Visualization metrics and titles
VIZ_METRICS = [
    ("loc", "Lines of Code (Raw)"),
    ("complexity_per_100_loc", "Complexity (Per 100 LOC)"),
    ("avg_line_len", "Avg Line Length"),
    ("comment_density", "Comment Density"),
    ("indent_depth", "Indentation Depth"),
    ("var_name_len", "Avg Variable Name Length")
]

### --- New: Statistical Testing Function ---
def run_mann_whitney_u_test(data, metric):
    """Performs Mann-Whitney U test between Human and AI distributions."""
    human_data = data[data['author_type'] == 'human'][metric].dropna()
    ai_data = data[data['author_type'] == 'ai'][metric].dropna()
    
    # Filter out zeros/extremes for cleaner testing if data size is sufficient
    # We'll use all non-NaN data for robustness.
    
    if len(human_data) < 20 or len(ai_data) < 20: # Arbitrary minimum sample size
        return "N/A (Insufficient Data)"

    # Mann-Whitney U test (non-parametric, ideal for comparing distributions not assumed to be normal)
    u_statistic, p_value = stats.mannwhitneyu(human_data, ai_data, alternative='two-sided')
    
    # We primarily care about the p-value
    if p_value < 0.001:
        return f"p < 0.001 (Highly Significant)"
    elif p_value < 0.05:
        return f"p = {p_value:.3f} (Significant)"
    else:
        return f"p = {p_value:.3f} (Not Significant)"

### --- Enhanced Visualization Functions ---

def plot_violin_vs_author_type(data, metric, title, filename):
    """Compares Human vs AI using a Violin Plot for better distribution visualization."""
    plt.figure(figsize=(8, 6))
    
    # Filter outliers for readable graphs
    q_high = data[metric].quantile(0.98)
    plot_data = data[data[metric] < q_high]

    sns.violinplot(
        data=plot_data, 
        x="author_type", 
        y=metric, 
        hue="author_type", 
        legend=False, 
        palette="Set2",
        inner="quartile", # Show median and quartiles inside the violin
    )
    
    # Run and display statistical test result
    p_result = run_mann_whitney_u_test(data, metric)
    
    plt.title(f"{title}\n(Human vs AI - Stat Test: {p_result})", fontweight='bold')
    plt.xlabel("Author Type")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close() 
    print(f"  - Saved {filename}")


def plot_violin_vs_model(data, metric, title, filename):
    """Compares all four sources (Human, Chat, Deepseek, Qwen) using a Violin Plot."""
    plt.figure(figsize=(10, 6))
    
    q_high = data[metric].quantile(0.98)
    plot_data = data[data[metric] < q_high]

    sns.violinplot(
        data=plot_data, 
        x="model", 
        y=metric, 
        palette=["#34a853", "#4285f4", "#ea4335", "#fbbc05"], # Google-like colors for clarity
        inner="quartile",
        order=['human', 'chatgpt', 'deepseek', 'qwen']
    )
    
    plt.title(f"{title} (Comparison by Specific Source)", fontweight='bold')
    plt.xlabel("Source Model")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close() 
    print(f"  - Saved {filename}")


def plot_umap(data, hue_col='author_type', filename="umap_clustering_authortype.png", title_suffix="Author Type"):
    """
    Generates a UMAP scatter plot, configurable by 'author_type' or 'model'.
    """
    if data.empty:
        print(f"\nSkipping UMAP plot ({hue_col}): Sample data is empty.")
        return
        
    plt.figure(figsize=(10, 8))
    
    # Define distinct color palettes for the two plots
    if hue_col == 'author_type':
        palette = {'human': 'darkorange', 'ai': 'dodgerblue'}
    else: # hue_col == 'model'
        # Human/AI Model Colors: Human is orange, AIs are shades of blue/green/red
        palette = {'human': '#FF9900', 'chatgpt': '#007ACC', 'deepseek': '#4CAF50', 'qwen': '#F44336'} 

    sns.scatterplot(
        x='UMAP 1', 
        y='UMAP 2', 
        hue=hue_col, 
        data=data, 
        palette=palette,
        s=20, 
        alpha=0.6,
    )
    plt.title(f'Code Style Clustering (UMAP) - Colored by {title_suffix} (Sampled Data)', fontweight='bold')
    plt.legend(title=title_suffix)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()
    print(f"  - Saved {filename}") 

# --- Main loop to generate visualizations ---

# 1. Violin Plots (Human vs AI) - Uses the full structural dataset (df_boxplot)
print("\n--- Generating Violin Plots (6 files, Human vs AI, Full Structural Dataset) ---")
for metric, title in VIZ_METRICS:
    filename = f"{metric.replace('_per_100_loc', '_norm').lower()}_violin_authortype.png"
    plot_violin_vs_author_type(df_boxplot, metric, title, filename)


# 2. Violin Plots (Comparison by Specific Model) - Uses the full structural dataset (df_boxplot)
print("\n--- Generating Violin Plots (6 files, Model Comparison, Full Structural Dataset) ---")
for metric, title in VIZ_METRICS:
    filename = f"{metric.replace('_per_100_loc', '_norm').lower()}_violin_model.png"
    plot_violin_vs_model(df_boxplot, metric, title, filename)


# 3. UMAP Plot (Author Type - Human vs AI) - Uses the sampled embedding dataset (df_umap)
print("\n--- Generating Embedding Plot (UMAP by Author Type) ---")
plot_umap(df_umap, hue_col='author_type', filename="umap_clustering_authortype.png", title_suffix="Author Type (Human vs AI)")


# 4. UMAP Plot (Model - Human vs 3 AIs) - Uses the sampled embedding dataset (df_umap)
print("\n--- Generating Embedding Plot (UMAP by Specific Model) ---")
plot_umap(df_umap, hue_col='model', filename="umap_clustering_model.png", title_suffix="Specific Model (Human vs 3 AIs)")

print("\nVisualization complete. 14 charts saved as individual PNG files.")