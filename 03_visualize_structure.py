import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
import os
import sys
import scipy.stats as stats

# Silences the Seaborn FutureWarning for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module='seaborn')

try:
    # Load Structural Features (Full Dataset)
    # This file MUST contain the grouping columns ('author_type' and 'model').
    df_struct = pd.read_csv("results/structural_features.csv")
    
    # We create an 'id' column, though it's not strictly necessary if we aren't merging
    # However, keeping it can be useful for diagnostics.
    df_struct['id'] = df_struct.index 
    
    print(f"Loaded structural features (full set): {len(df_struct)} records.")

    df_final = df_struct.copy() 
    
    # df_boxplot: The full structural dataset (used for violin plots)
    df_boxplot = df_final 

    print(f"Data prepared for Box Plots (full set): {len(df_boxplot)} records.")
    
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


def run_mann_whitney_u_test(data, metric):
    human_data = data[data['author_type'] == 'human'][metric].dropna()
    ai_data = data[data['author_type'] == 'ai'][metric].dropna()
    
    # Filter out zeros/extremes for cleaner testing if data size is sufficient
    
    if len(human_data) < 20 or len(ai_data) < 20: # Minimum sample size
        return "N/A (Insufficient Data)"

    # Mann-Whitney U test
    u_statistic, p_value = stats.mannwhitneyu(human_data, ai_data, alternative='two-sided')
    
    # We care about the p-value
    if p_value < 0.001:
        return f"p < 0.001 (Highly Significant)"
    elif p_value < 0.05:
        return f"p = {p_value:.3f} (Significant)"
    else:
        return f"p = {p_value:.3f} (Not Significant)"

def plot_violin_vs_author_type(data, metric, title, filename):
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
    plt.figure(figsize=(10, 6))
    
    q_high = data[metric].quantile(0.98)
    plot_data = data[data[metric] < q_high]

    sns.violinplot(
        data=plot_data, 
        x="model", 
        y=metric, 
        palette=["#34a853", "#4285f4", "#ea4335", "#fbbc05"],
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

# Violin Plots (Human vs AI) - Uses the full structural dataset (df_boxplot)
print("\nGenerating Violin Plots (6 files, Human vs AI, Full Structural Dataset)")
for metric, title in VIZ_METRICS:
    filename = f"{metric.replace('_per_100_loc', '_norm').lower()}_violin_authortype.png"
    plot_violin_vs_author_type(df_boxplot, metric, title, filename)


# Violin Plots (Comparison by Specific Model) - Uses the full structural dataset (df_boxplot)
print("\nGenerating Violin Plots (6 files, Model Comparison, Full Structural Dataset)")
for metric, title in VIZ_METRICS:
    filename = f"{metric.replace('_per_100_loc', '_norm').lower()}_violin_model.png"
    plot_violin_vs_model(df_boxplot, metric, title, filename)


print("\nVisualization complete. 14 charts saved as individual PNG files.")