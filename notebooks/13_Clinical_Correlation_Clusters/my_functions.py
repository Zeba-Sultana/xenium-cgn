import os
import scanpy as sc
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

out_dir = "output"

def plot_correlation_scatter(
    df, 
    x_col, 
    y_col, 
    label_pos=[0.65, 0.9],
    figsize=(4.5, 4.5),
    method='pearson', 
    output_file='scatter_plot.png',
    best_fit=False  # New argument

):
    """
    Plots a scatter plot and computes correlation between two specified columns.

    Parameters:
    - df: pandas DataFrame
    - x_col: str, name of the first column (X-axis)
    - y_col: str, name of the second column (Y-axis)
    - method: str, 'pearson' or 'spearman'
    - output_file: str, filename to save the plot
    """

    # Filter rows where y_col is not 'na'
    df_filtered = df[df[y_col] != 'na'].copy()

    # Convert columns to numeric, coerce errors
    df_filtered[x_col] = pd.to_numeric(df_filtered[x_col], errors='coerce')
    df_filtered[y_col] = pd.to_numeric(df_filtered[y_col], errors='coerce')

    # Drop rows with NaNs in either column
    df_filtered = df_filtered.dropna(subset=[x_col, y_col])

    # Choose correlation method
    if method.lower() == 'pearson':
        corr_func = pearsonr
        corr_name = 'Pearson'
    elif method.lower() == 'spearman':
        corr_func = spearmanr
        corr_name = 'Spearman'
    else:
        raise ValueError("Method must be 'pearson' or 'spearman'.")

    correlation, p_value = corr_func(df_filtered[x_col], df_filtered[y_col])

    # Plot
    plt.figure(figsize=figsize)
    plt.scatter(
        df_filtered[x_col], 
        df_filtered[y_col], 
        alpha=0.7, 
        s=50, 
        edgecolor='k'
    )
    
    if best_fit:
        # Fit a linear regression line
        fit = np.polyfit(df_filtered[x_col], df_filtered[y_col], 1)
        fit_fn = np.poly1d(fit)
        x_sorted = np.sort(df_filtered[x_col])
        plt.plot(x_sorted, fit_fn(x_sorted), color='grey', linestyle='--', linewidth=2, label='Best Fit Line')
    
    plt.title(f"{x_col} vs {y_col}", fontsize=14)
    plt.xlabel(x_col, fontsize=14)
    plt.ylabel(y_col, fontsize=14)

    # Annotate correlation
    plt.text(
        label_pos[0], 
        label_pos[1],
        f"{corr_name} corr.: {correlation:.2f}\np-value: {p_value:.2e}",
        fontsize=10,
        ha='left',
        va='top',
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round,pad=0.5", edgecolor="none", facecolor="none")
    )

    #plt.grid(visible=False, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    output_file = os.path.join(out_dir, f"{x_col}_vs_{y_col}_{method.lower()}.pdf")
    output_file2 = os.path.join(out_dir, f"{x_col}_vs_{y_col}_{method.lower()}.png")

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')

    plt.show()
