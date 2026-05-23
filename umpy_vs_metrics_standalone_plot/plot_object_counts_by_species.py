"""
Standalone script to create the Object Counts vs Bond by Species with UMPY Relative plot from CSV data.

This script:
1. Reads the object_counts_vs_bond_by_species_with_umpy_relative_data.csv file
2. Creates a 2x2 subplot showing:
   - Three plots (S, Su, Sns) with active satellites, compliant derelicts, and non-compliant derelicts
   - One plot showing relative change in final year UMPY from baseline (%) vs bond amount
3. Saves the plot as object_counts_vs_bond_by_species_with_umpy_relative.png

Usage:
    python plot_object_counts_by_species.py

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
    Create 2x2 subplot from CSV data showing object counts by species and UMPY relative change.
    
    Args:
        csv_path: Path to the CSV file
        output_path: Path to save the plot (default: same directory as CSV)
    """
    # Read CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Separate data by PMD lifetime and species
    species_names = ["S", "Su", "Sns"]
    
    # Consistent colors across all graphs
    active_color = "blue"
    compliant_color = "green"
    non_compliant_color = "red"
    umpy_relative_color = "purple"
    
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    # Plot first three subplots for S, Su, Sns
    for idx, species_key in enumerate(species_names):
        ax = axes[idx]
        
        # Get data for this species
        species_df = df[df['species'] == species_key].copy()
        
        # Separate by lifetime
        df_5yr = species_df[species_df['pmd_lifetime_years'] == 5].sort_values('bond_amount_millions')
        df_25yr = species_df[species_df['pmd_lifetime_years'] == 25].sort_values('bond_amount_millions')
        
        # Plot 5yr PMD (solid lines) - only if we have data
        if not df_5yr.empty and df_5yr['active_satellites'].notna().any():
            active_5yr = df_5yr[df_5yr['active_satellites'].notna()]
            ax.plot(
                active_5yr['bond_amount_millions'], 
                active_5yr['active_satellites'], 
                color=active_color, 
                marker='s', 
                linestyle='-', 
                linewidth=2, 
                markersize=6, 
                label=f'Active {species_key} Satellites (5yr)', 
                zorder=3
            )
        
        if not df_5yr.empty and df_5yr['compliant_derelicts'].notna().any():
            compliant_5yr = df_5yr[df_5yr['compliant_derelicts'].notna()]
            ax.plot(
                compliant_5yr['bond_amount_millions'], 
                compliant_5yr['compliant_derelicts'], 
                color=compliant_color, 
                marker='o', 
                linestyle='-', 
                linewidth=2, 
                markersize=6, 
                label='Compliant Derelicts (5yr)', 
                zorder=2
            )
        
        if not df_5yr.empty and df_5yr['non_compliant_derelicts'].notna().any():
            non_compliant_5yr = df_5yr[df_5yr['non_compliant_derelicts'].notna()]
            ax.plot(
                non_compliant_5yr['bond_amount_millions'], 
                non_compliant_5yr['non_compliant_derelicts'], 
                color=non_compliant_color, 
                marker='x', 
                linestyle='-', 
                linewidth=2, 
                markersize=6, 
                label='Non-Compliant Derelicts (5yr)', 
                zorder=2
            )
        
        # Plot 25yr PMD (dashed lines) - only if we have data
        if not df_25yr.empty and df_25yr['active_satellites'].notna().any():
            active_25yr = df_25yr[df_25yr['active_satellites'].notna()]
            ax.plot(
                active_25yr['bond_amount_millions'], 
                active_25yr['active_satellites'], 
                color=active_color, 
                marker='s', 
                linestyle='--', 
                linewidth=2, 
                markersize=6, 
                label=f'Active {species_key} Satellites (25yr)', 
                zorder=3, 
                alpha=0.8
            )
        
        if not df_25yr.empty and df_25yr['compliant_derelicts'].notna().any():
            compliant_25yr = df_25yr[df_25yr['compliant_derelicts'].notna()]
            ax.plot(
                compliant_25yr['bond_amount_millions'], 
                compliant_25yr['compliant_derelicts'], 
                color=compliant_color, 
                marker='o', 
                linestyle='--', 
                linewidth=2, 
                markersize=6, 
                label='Compliant Derelicts (25yr)', 
                zorder=2, 
                alpha=0.8
            )
        
        if not df_25yr.empty and df_25yr['non_compliant_derelicts'].notna().any():
            non_compliant_25yr = df_25yr[df_25yr['non_compliant_derelicts'].notna()]
            ax.plot(
                non_compliant_25yr['bond_amount_millions'], 
                non_compliant_25yr['non_compliant_derelicts'], 
                color=non_compliant_color, 
                marker='x', 
                linestyle='--', 
                linewidth=2, 
                markersize=6, 
                label='Non-Compliant Derelicts (25yr)', 
                zorder=2, 
                alpha=0.8
            )
        
        ax.set_xlabel("Lifetime Bond Amount, $ (M)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Number of Objects", fontsize=14, fontweight='bold')
        ax.set_title(f"{species_key}", fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        
        # Make ticks bold
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.5)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
    
    # Plot fourth subplot for relative change in final year UMPY
    ax = axes[3]
    
    # Get UMPY data (same for all species, so just get from first species)
    umpy_df = df[df['species'] == 'S'].copy()  # Use S as representative
    
    # Separate by lifetime
    umpy_5yr = umpy_df[umpy_df['pmd_lifetime_years'] == 5].sort_values('bond_amount_millions')
    umpy_25yr = umpy_df[umpy_df['pmd_lifetime_years'] == 25].sort_values('bond_amount_millions')
    
    # Plot 5yr UMPY relative change
    if not umpy_5yr.empty and umpy_5yr['umpy_percent_change'].notna().any():
        umpy_5yr_clean = umpy_5yr[umpy_5yr['umpy_percent_change'].notna()]
        ax.plot(
            umpy_5yr_clean['bond_amount_millions'], 
            umpy_5yr_clean['umpy_percent_change'], 
            color=umpy_relative_color, 
            marker='^', 
            linestyle='-', 
            linewidth=2, 
            markersize=6, 
            label='Relative Change in UMPY (5yr)', 
            zorder=3
        )
    
    # Plot 25yr UMPY relative change
    if not umpy_25yr.empty and umpy_25yr['umpy_percent_change'].notna().any():
        umpy_25yr_clean = umpy_25yr[umpy_25yr['umpy_percent_change'].notna()]
        ax.plot(
            umpy_25yr_clean['bond_amount_millions'], 
            umpy_25yr_clean['umpy_percent_change'], 
            color=umpy_relative_color, 
            marker='^', 
            linestyle='--', 
            linewidth=2, 
            markersize=6, 
            label='Relative Change in UMPY (25yr)', 
            zorder=3, 
            alpha=0.8
        )
    
    # Add horizontal line at 0% for baseline
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
    
    ax.set_xlabel("Lifetime Bond Amount, $ (M)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Relative Change in UMPY (%)", fontsize=14, fontweight='bold')
    ax.set_title("Relative UMPY difference to Baseline", fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13, loc='upper right')
    
    # Make ticks bold and increase size by 2
    ax.tick_params(axis='both', which='major', labelsize=13, width=1.5)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = os.path.join(os.path.dirname(csv_path), "object_counts_vs_bond_by_species_with_umpy_relative.png")
    
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
    csv_path = os.path.join(script_dir, "object_counts_vs_bond_by_species_with_umpy_relative_data.csv")
    output_path = os.path.join(script_dir, "object_counts_vs_bond_by_species_with_umpy_relative.png")
    
    print(f"Reading CSV from: {csv_path}")
    create_plot_from_csv(csv_path, output_path)
    print("Plot creation complete!")


if __name__ == "__main__":
    main()


