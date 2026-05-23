"""
Standalone script to create the UMPY vs Collision Probability plot from CSV data.

This script:
1. Reads the combined_umpy_vs_metrics_data.csv file
2. Creates a scatter plot showing:
   - Final UMPY vs Relative Change in Collision Probability (%)
3. Saves the plot as combined_umpy_vs_metrics.png

Usage:
    python plot_umpy_vs_collision_probability.py

Requirements:
    - pandas
    - matplotlib
    - numpy
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def format_bond_label(bond_millions):
    """
    Format bond amount for display: Baseline, $0.1M, $2.0M, etc.
    
    Args:
        bond_millions: Bond amount in millions of dollars
    
    Returns:
        str: Formatted label
    """
    if bond_millions == 0:
        return "Baseline"
    elif bond_millions == int(bond_millions):
        # Whole number: $1M, $2M, etc.
        return f"${int(bond_millions)}.0M"
    else:
        # Decimal: $0.1M, $0.5M, $1.5M, etc.
        return f"${bond_millions:.1f}M"


def create_plot_from_csv(csv_path, output_path=None):
    """
    Create scatter plot from CSV data showing UMPY vs relative change in collision probability.
    
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
    
    # Calculate limits
    all_x = df['final_umpy'].values
    all_y = df['relative_change_percent'].values
    
    xmin_raw, xmax_raw = min(all_x), max(all_x)
    ymin_raw, ymax_raw = min(all_y), max(all_y)
    
    def calc_limits(min_val, max_val):
        range_val = max_val - min_val if max_val != min_val else 1.0
        margin = 0.10 * range_val
        return min_val - margin, max_val + margin
    
    xmin, xmax = calc_limits(xmin_raw, xmax_raw)
    ymin, ymax = calc_limits(ymin_raw, ymax_raw)
    
    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Colorblind-friendly colors: blue and orange
    colour_map = {"Start of Disposal Orbit": "#1f77b4", "End of Disposal Orbit (Deorbited)": "#ff7f0e"}
    markers = ["s", "o"]
    
    for i, sim_name in enumerate(unique_sims):
        sim_data = df[df['simulation_name'] == sim_name]
        
        display_name = name_mapping.get(sim_name, sim_name)
        colour = colour_map.get(display_name, "black")
        marker = markers[i % len(markers)]
        
        umpy = sim_data['final_umpy'].values
        relative_changes = sim_data['relative_change_percent'].values
        bond_amounts = sim_data['bond_amount_millions'].values
        
        # Plot: UMPY vs Relative Change in Collision Probability
        # - Baseline point(s) are rendered in black (both point + label).
        # Use a tolerance check to avoid float-equality issues.
        is_baseline = np.isclose(bond_amounts, 0.0, atol=1e-9)

        # Plot non-baseline points first (series color)
        if np.any(~is_baseline):
            ax.scatter(
                umpy[~is_baseline],
                relative_changes[~is_baseline],
                s=150,
                marker=marker,
                color=colour,
                alpha=0.85,
                edgecolors='none',  # Remove stroke/edge
                linewidths=0,
                label=display_name,
                zorder=3,
            )

        # Plot baseline point(s) last so they can't be overpainted
        if np.any(is_baseline):
            ax.scatter(
                umpy[is_baseline],
                relative_changes[is_baseline],
                s=170,
                marker=marker,
                color='black',
                alpha=0.95,
                edgecolors='none',  # Remove stroke/edge
                linewidths=0,
                label=None,  # keep legend tied to disposal type, not baseline
                zorder=5,
            )
        
        # Add bond amount labels to each point
        for x, y, bond_millions in zip(umpy, relative_changes, bond_amounts):
            label = format_bond_label(bond_millions)
            
            # Use black color for Baseline label, otherwise use point color
            label_color = 'black' if label == 'Baseline' else colour
            
            # Determine label position: 
            # End of Disposal Orbit (Deorbited) goes below, Start of Disposal Orbit goes above
            place_below = (display_name == "End of Disposal Orbit (Deorbited)")
            
            # Special handling for $1.0M labels
            if bond_millions == 1.0:
                if display_name == "Start of Disposal Orbit":
                    # Red $1.0M - move slightly to top left (1000 UMPY to the left, small amount above)
                    y_range = ymax - ymin
                    ax.annotate(
                        label,
                        xy=(x, y),
                        xytext=(x - 1000, y + y_range * 0.01),  # 1000 UMPY left, small amount above
                        textcoords='data',
                        ha='right',  # Right align (so text extends left from point)
                        va='bottom',  # Bottom align vertically
                        fontsize=10,
                        fontweight='bold',
                        color=label_color,
                        alpha=0.8,
                        zorder=4,
                    )
                else:
                    # Green $1.0M (End of Disposal Orbit) - put at the top
                    ax.annotate(
                        label,
                        xy=(x, y),
                        xytext=(0, 15),  # Offset: 0 horizontal (centered), 15 points above
                        textcoords='offset points',
                        ha='center',  # Center align horizontally
                        fontsize=10,
                        fontweight='bold',
                        color=label_color,
                        alpha=0.8,
                        zorder=4,
                    )
            elif place_below:
                # Place label below the point (End of Disposal Orbit)
                ax.annotate(
                    label,
                    xy=(x, y),
                    xytext=(0, -15),  # Offset: 0 horizontal (centered), 15 points below
                    textcoords='offset points',
                    ha='center',  # Center align horizontally
                    fontsize=10,
                    fontweight='bold',
                    color=label_color,
                    alpha=0.8,
                    zorder=4,
                )
            else:
                # Place label above the point (Start of Disposal Orbit)
                ax.annotate(
                    label,
                    xy=(x, y),
                    xytext=(0, 10),  # Offset: 0 horizontal (centered), 10 points above
                    textcoords='offset points',
                    ha='center',  # Center align horizontally
                    fontsize=10,
                    fontweight='bold',
                    color=label_color,
                    alpha=0.8,
                    zorder=4,
                )
    
    # Add horizontal line at 0% (baseline)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
    
    # Set labels and limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Final UMPY (kg/year)", fontsize=16, fontweight="black")
    ax.set_ylabel("Relative Change in Collision Probability (%)", fontsize=16, fontweight="black")
    
    # Make axis ticks bold and bigger
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    ax.grid(True, alpha=0.3)
    ax.legend(title="Bond Refunded at", fontsize=14, frameon=True, loc="upper left")
    if ax.get_legend():
        ax.get_legend().get_title().set_fontweight("black")
        ax.get_legend().get_title().set_fontsize(16)
        for text in ax.get_legend().get_texts():
            text.set_fontweight("black")
            text.set_horizontalalignment('left')
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = os.path.join(os.path.dirname(csv_path), "combined_umpy_vs_metrics.png")
    
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
    output_path = os.path.join(script_dir, "combined_umpy_vs_metrics.png")
    
    print(f"Reading CSV from: {csv_path}")
    create_plot_from_csv(csv_path, output_path)
    print("Plot creation complete!")


if __name__ == "__main__":
    main()

