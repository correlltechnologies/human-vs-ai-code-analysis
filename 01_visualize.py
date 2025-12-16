# Save this as 03_visualize.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
import os
import sys

# Silences the Seaborn FutureWarning for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module='seaborn')

# --- 1. Load Data ---

try:
    df = pd.read_csv("results/analysis_data.csv")
    print(f"Data loaded successfully from results/analysis_data.csv: {len(df)} records.")
except FileNotFoundError:
    print("\nERROR: 'results/analysis_data.csv' not found.")
    print("Please run 01_extract_features.py AND 02_generate_embeddings.py first.")
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

# Plotting functions
def plot_boxplot(data, metric, title, filename):
    plt.figure(figsize=(7, 5))
    
    # Filter outliers (98th percentile) for readable graphs
    q_high = data[metric].quantile(0.98)
    plot_data = data[data[metric] < q_high]

    sns.boxplot(
        data=plot_data, 
        x="author_type", 
        y=metric, 
        hue="author_type", 
        legend=False, 
        palette="Set2"
    )
    plt.title(title, fontweight='bold')
    plt.xlabel("Author Type")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close() # Close plot to free memory
    print(f"  - Saved {filename}")


# Generate umap scatter plot
def plot_umap(data, filename="umap_clustering.png"):
    plt.figure(figsize=(10, 8))
    
    # Ensure the UMAP plotting data is correctly prepared
    plot_df = data[['UMAP 1', 'UMAP 2', 'author_type']].rename(columns={'author_type': 'Author Type'})

    sns.scatterplot(
        x='UMAP 1', 
        y='UMAP 2', 
        hue='Author Type', 
        data=plot_df, 
        palette={'human': 'darkorange', 'ai': 'dodgerblue'},
        s=20, 
        alpha=0.6,
    )
    plt.title('Code Style Clustering using CodeBERT Embeddings (UMAP)', fontweight='bold')
    plt.legend(title='Author Type')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()
    print(f"  - Saved {filename}") 

# --- Main loop to generate visualizations ---

# Box Plots
print("\n--- Generating Box Plots (6 files) ---")
for metric, title in VIZ_METRICS:
    filename = f"{metric.replace('_per_100_loc', '_norm').lower()}_boxplot.png"
    plot_boxplot(df, metric, title, filename)


# UMAP Plot
print("\n--- Generating Embedding Plot (1 file) ---")
plot_umap(df)

print("\nVisualization complete. All 7 charts saved as individual PNG files.")