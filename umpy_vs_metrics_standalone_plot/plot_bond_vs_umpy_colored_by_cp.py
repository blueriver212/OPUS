"""
Standalone script to create a Bond vs UMPY plot colored by Collision Probability change.

This script:
1. Reads the combined_umpy_vs_metrics_data.csv file
2. Creates a scatter plot showing:
   - X-axis: Bond Amount ($ millions)
   - Y-axis: Final UMPY (kg/year)
   - Color: Relative Change in Collision Probability (%)
3. Saves the plot as bond_vs_umpy_colored_by_cp.png

Usage:
    python plot_bond_vs_umpy_colored_by_cp.py

Requirements:
    - pandas
    - matplotlib
    - numpy
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def create_plot_from_csv(csv_path, output_path=None):
    """
    Create scatter plot from CSV data showing Bond vs UMPY, colored by collision probability change.
    
    Args:
        csv_path: Path to the CSV file
        output_path: Path to save the plot (default: same directory as CSV)
    """
    # Read CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Name mapping for display
    name_mapping = {
        'extensive_new': 'Start of Disposal Orbit',
        'intensive': 'End of Disposal Orbit (Deorbited)'
    }
    
    # Get unique simulation names
    unique_sims = df['simulation_name'].unique()
    
    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Get color range for collision probability change
    cp_min = df['relative_change_percent'].min()
    cp_max = df['relative_change_percent'].max()
    
    # Create color map (using a colormap that works well for negative values)
    # Using 'RdYlGn_r' reversed so more negative (better) is greener, less negative is redder
    
    # Colorblind-friendly colors for series lines
    series_colors = {
        'Start of Disposal Orbit': '#1f77b4',  # Blue
        'End of Disposal Orbit (Deorbited)': '#ff7f0e'  # Orange
    }
    
    # Collect all scatter plots to create a single colorbar
    scatter_plots = []
    
    for sim_name in unique_sims:
        sim_data = df[df['simulation_name'] == sim_name].copy()
        
        display_name = name_mapping.get(sim_name, sim_name)
        
        # Sort by bond amount for connecting lines
        sim_data = sim_data.sort_values('bond_amount_millions')
        
        bond_amounts = sim_data['bond_amount_millions'].values
        umpy = sim_data['final_umpy'].values
        cp_changes = sim_data['relative_change_percent'].values
        
        # Use different markers for different simulation types
        marker = 's' if display_name == 'Start of Disposal Orbit' else 'o'
        series_color = series_colors.get(display_name, 'gray')
        
        # Add line connecting points for this series
        ax.plot(
            bond_amounts,
            umpy,
            color=series_color,
            linestyle='-',
            linewidth=2,
            alpha=0.5,
            zorder=1,
            label=display_name
        )
        
        # Create scatter plot with color based on collision probability change
        scatter = ax.scatter(
            bond_amounts,
            umpy,
            c=cp_changes,
            s=150,
            marker=marker,
            cmap='RdYlGn_r',  # Reversed: green for more negative (better), red for less negative
            vmin=cp_min,
            vmax=cp_max,
            alpha=0.85,
            edgecolors='none',
            linewidths=0,
            zorder=3,
        )
        scatter_plots.append(scatter)
    
    # Add colorbar (use the last scatter plot for colorbar)
    if scatter_plots:
        cbar = plt.colorbar(scatter_plots[-1], ax=ax)
        cbar.set_label('Relative Change in Collision Probability (%)', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight('bold')
    
    # Set labels
    ax.set_xlabel("Bond Amount ($ millions)", fontsize=16, fontweight="black")
    ax.set_ylabel("Final UMPY (kg/year)", fontsize=16, fontweight="black")
    
    # Make axis ticks bold and bigger
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    ax.grid(True, alpha=0.3)
    ax.legend(title="Bond Refunded at", fontsize=14, frameon=True, loc="best")
    if ax.get_legend():
        ax.get_legend().get_title().set_fontweight("black")
        ax.get_legend().get_title().set_fontsize(16)
        for text in ax.get_legend().get_texts():
            text.set_fontweight("black")
            text.set_horizontalalignment('left')
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = os.path.join(os.path.dirname(csv_path), "bond_vs_umpy_colored_by_cp.png")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    """
    Main function to create plot from CSV.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "combined_umpy_vs_metrics_data.csv")
    output_path = os.path.join(script_dir, "bond_vs_umpy_colored_by_cp.png")
    
    print(f"Reading CSV from: {csv_path}")
    create_plot_from_csv(csv_path, output_path)
    print("Plot creation complete!")


if __name__ == "__main__":
    main()

