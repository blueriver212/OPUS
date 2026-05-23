"""
Standalone script to create the Welfare and UMPY % Change plot from CSV data.

This script:
1. Reads the total_satellites_and_umpy_percent_change_data.csv file
2. Creates a single plot showing:
   - Welfare % Change vs Bond Amount (5yr and 25yr)
   - UMPY % Change vs Bond Amount (5yr and 25yr)
3. Saves the plot as total_satellites_and_umpy_percent_change.png

Usage:
    python plot_from_csv.py

Requirements:
    - pandas
    - matplotlib
    - numpy
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_plot_from_csv(csv_path, output_path=None):
    """
    Create plot from CSV data showing Welfare and UMPY % change vs bond amount.
    
    Args:
        csv_path: Path to the CSV file
        output_path: Path to save the plot (default: same directory as CSV)
    """
    # Read CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Separate data by PMD lifetime
    df_5yr = df[df['pmd_lifetime_years'] == 5].copy()
    df_25yr = df[df['pmd_lifetime_years'] == 25].copy()
    
    # Sort by bond amount
    df_5yr = df_5yr.sort_values('bond_amount_millions')
    df_25yr = df_25yr.sort_values('bond_amount_millions')
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Colors - red/dark red for UMPY, blue/dark blue for Welfare (5yr/25yr coupling)
    welfare_color_5yr = "blue"
    welfare_color_25yr = "darkblue"
    umpy_color_5yr = "red"
    umpy_color_25yr = "darkred"
    
    # Single marker style for all lines
    marker_style = 'o'
    
    # Plot Welfare (5yr - solid)
    if not df_5yr.empty and df_5yr['satellites_percent_change'].notna().any():
        welfare_5yr = df_5yr[df_5yr['satellites_percent_change'].notna()]
        ax.plot(
            welfare_5yr['bond_amount_millions'], 
            welfare_5yr['satellites_percent_change'], 
            color=welfare_color_5yr, 
            marker=marker_style, 
            linestyle='-', 
            linewidth=2, 
            markersize=6, 
            label='Welfare - 5yr', 
            zorder=3
        )
    
    # Plot Welfare (25yr - solid)
    if not df_25yr.empty and df_25yr['satellites_percent_change'].notna().any():
        welfare_25yr = df_25yr[df_25yr['satellites_percent_change'].notna()]
        ax.plot(
            welfare_25yr['bond_amount_millions'], 
            welfare_25yr['satellites_percent_change'], 
            color=welfare_color_25yr, 
            marker=marker_style, 
            linestyle='-', 
            linewidth=2, 
            markersize=6, 
            label='Welfare - 25yr', 
            zorder=3
        )
    
    # Plot UMPY (5yr - solid)
    if not df_5yr.empty and df_5yr['umpy_percent_change'].notna().any():
        umpy_5yr = df_5yr[df_5yr['umpy_percent_change'].notna()]
        ax.plot(
            umpy_5yr['bond_amount_millions'], 
            umpy_5yr['umpy_percent_change'], 
            color=umpy_color_5yr, 
            marker=marker_style, 
            linestyle='-', 
            linewidth=2, 
            markersize=6, 
            label='UMPY - 5yr', 
            zorder=3
        )
    
    # Plot UMPY (25yr - solid)
    if not df_25yr.empty and df_25yr['umpy_percent_change'].notna().any():
        umpy_25yr = df_25yr[df_25yr['umpy_percent_change'].notna()]
        ax.plot(
            umpy_25yr['bond_amount_millions'], 
            umpy_25yr['umpy_percent_change'], 
            color=umpy_color_25yr, 
            marker=marker_style, 
            linestyle='-', 
            linewidth=2, 
            markersize=6, 
            label='UMPY - 25yr', 
            zorder=3
        )
    
    # Add horizontal line at 0% for baseline
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
    
    ax.set_xlabel("Lifetime Bond Amount, $ (M)", fontsize=18, fontweight='bold')
    ax.set_ylabel("Relative Change to Baseline (%)", fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=16, loc='best')
    
    # Make ticks bold and larger
    ax.tick_params(axis='both', which='major', labelsize=16, width=1.5)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = os.path.join(os.path.dirname(csv_path), "total_satellites_and_umpy_percent_change.png")
    
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
    csv_path = os.path.join(script_dir, "total_satellites_and_umpy_percent_change_data.csv")
    output_path = os.path.join(script_dir, "total_satellites_and_umpy_percent_change.png")
    
    print(f"Reading CSV from: {csv_path}")
    create_plot_from_csv(csv_path, output_path)
    print("Plot creation complete!")


if __name__ == "__main__":
    main()
