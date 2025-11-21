"""
Script to combine scatter_noncompliance_vs_bond plots from multiple simulations.

This script:
1. Loads data from multiple simulation folders
2. Extracts raw data (bond amount, non-compliance, total money paid)
3. Combines all data points
4. Creates a combined scatter plot
5. Saves raw data to CSV for further analysis
"""

import os
import sys
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add OPUS directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'OPUS'))

from utils.PlotHandler import PlotData
from utils.MultiSpecies import MultiSpecies
from utils.MocatParameters import configure_mocat


def extract_bond_from_scenario(scenario_name):
    """Extract bond amount from scenario name."""
    match = re.search(r"bond_(\d+)k", scenario_name.lower())
    if match:
        return int(match.group(1)) * 1_000
    elif 'baseline' in scenario_name.lower():
        return 0
    else:
        return 0


def load_simulation_data(simulation_name, scenario_files, MOCAT):
    """
    Load data from a simulation folder.
    
    Returns:
        list of tuples: (bond_vals, noncompliance_vals, total_money_vals, scenario_names)
    """
    simulation_folder = os.path.join("Results", simulation_name)
    
    if not os.path.exists(simulation_folder):
        print(f"Warning: {simulation_folder} does not exist. Skipping...")
        return []
    
    bond_vals = []
    noncompliance_vals = []
    total_money_vals = []
    scenario_names = []
    
    for scenario in scenario_files:
        scenario_folder = os.path.join(simulation_folder, scenario)
        
        if not os.path.exists(scenario_folder):
            print(f"Warning: {scenario_folder} does not exist. Skipping...")
            continue
        
        try:
            # Load PlotData
            plot_data = PlotData(scenario, scenario_folder, MOCAT)
            other_data = plot_data.get_other_data()
            
            if other_data is None:
                print(f"Warning: No other_data found for {scenario}. Skipping...")
                continue
            
            # Get final timestep
            timesteps = sorted(other_data.keys(), key=int)
            if not timesteps:
                print(f"Warning: No timesteps found for {scenario}. Skipping...")
                continue
            
            final_timestep = timesteps[-1]
            nc = other_data[final_timestep].get("non_compliance", {})
            
            # Sum all non-compliance values if it's a dictionary
            if isinstance(nc, dict):
                nc_total = sum(nc.values())
            else:
                nc_total = nc if nc is not None else 0
            
            # Extract bond amount
            bond = extract_bond_from_scenario(scenario)
            
            # Calculate total money paid
            total_money = bond * nc_total
            
            bond_vals.append(bond)
            noncompliance_vals.append(nc_total)
            total_money_vals.append(total_money)
            scenario_names.append(f"{simulation_name}/{scenario}")
            
        except Exception as e:
            print(f"Error loading {scenario}: {e}")
            continue
    
    return bond_vals, noncompliance_vals, total_money_vals, scenario_names


def create_combined_plot(all_data, output_path=None):
    if output_path is None:
        output_path = "Results/combined_scatter_noncompliance_vs_bond.png"

    name_mapping = {
        'pmd_test': 'Extensive',
        'pmd_test_intensive': 'Intensive'
    }

    # ---------- first pass for limits ----------
    all_x, all_y = [], []
    for data in all_data:
        bonds = data['bonds']
        noncompliance = data['noncompliance']
        bonds_millions = [b / 1_000_000 for b in bonds]
        all_x.extend(bonds_millions)
        all_y.extend(noncompliance)

    xmin_raw, xmax_raw = min(all_x), max(all_x)
    ymin_raw, ymax_raw = min(all_y), max(all_y)

    x_range = xmax_raw - xmin_raw if xmax_raw != xmin_raw else 1.0
    y_range = ymax_raw - ymin_raw if ymax_raw != ymin_raw else 1.0

    x_margin = 0.10 * x_range
    y_margin = 0.10 * y_range

    xmin = xmin_raw - x_margin
    xmax = xmax_raw + x_margin
    ymin = ymin_raw - y_margin
    ymax = ymax_raw + y_margin

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    colour_map = {"Extensive": "red", "Intensive": "green"}
    markers = ["s", "o"]

    x_edge = 0.03 * (xmax - xmin)
    y_edge = 0.03 * (ymax - ymin)

    for i, data in enumerate(all_data):
        bonds = data['bonds']
        noncompliance = data['noncompliance']
        total_money = data['total_money']
        sim_name = data['simulation_name']

        display_name = name_mapping.get(sim_name, sim_name)
        colour = colour_map.get(display_name, "black")
        bonds_millions = [b / 1_000_000 for b in bonds]

        ax.scatter(
            bonds_millions,
            noncompliance,
            s=150,
            marker=markers[i % len(markers)],
            color=colour,
            alpha=0.85,
            edgecolors="k",
            linewidths=1.5,
            label=display_name,
            zorder=3,
        )

        # ---- annotations ----
        for idx, (x, y, total) in enumerate(
            zip(bonds_millions, noncompliance, total_money)
        ):
            label = f"£{round(total / 1_000_000):,}M"

            # base offsets
            if display_name == "Extensive":
                ox, oy = -55, -28  # bottom-left

                # SPECIAL CASE: move the 3rd Extensive label so it’s not on top of a point
                if idx == 2:
                    ox, oy = -55, 18  # above-left instead of below-left
            else:
                ox, oy = 14, 12  # top-right

            # flip near edges to keep labels inside grid
            if x < xmin + x_edge:
                ox = abs(ox)
            if x > xmax - x_edge:
                ox = -abs(ox)
            if y < ymin + y_edge:
                oy = abs(oy)
            if y > ymax - y_edge:
                oy = -abs(oy)

            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(ox, oy),
                fontsize=15,
                fontweight="black",
                color=colour,
                alpha=0.9,
            )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Bond Fee (£, million)", fontsize=18, fontweight="black")
    ax.set_ylabel("Final Derelict Count", fontsize=18, fontweight="black")

    leg = ax.legend(
        title="Disposal Type", fontsize=16, frameon=True, loc="upper right"
    )
    leg.get_title().set_fontweight("black")
    leg.get_title().set_fontsize(20)
    for text in leg.get_texts():
        text.set_fontweight("black")

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Combined scatter plot saved to {output_path}")
    plt.close()

def save_raw_data_to_csv(all_data, output_path=None):
    """
    Save raw data to CSV for further analysis.
    
    Args:
        all_data: List of dicts with simulation data
        output_path: Path to save CSV (default: Results/combined_scatter_data.csv)
    """
    if output_path is None:
        output_path = "Results/combined_scatter_data.csv"
    
    rows = []
    for data in all_data:
        for bond, nc, money, scenario in zip(
            data['bonds'], 
            data['noncompliance'], 
            data['total_money'],
            data['scenarios']
        ):
            rows.append({
                'simulation_name': data['simulation_name'],
                'scenario': scenario,
                'bond_amount': bond,
                'non_compliance_count': nc,
                'total_money_paid': money
            })
    
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Raw data saved to {output_path}")
    return df


def main():
    """
    Main function to combine scatter plots from multiple simulations.
    
    Edit the simulation_configs list to specify which simulations to combine.
    
    Usage:
        python combine_scatter_plots.py
    
    The script will:
        1. Load data from each simulation folder specified
        2. Extract bond amounts, non-compliance counts, and total money paid
        3. Create a combined scatter plot
        4. Save raw data to CSV
    """
    import json
    
    # Configuration: List of simulations to combine
    # Each entry should have: simulation_name, scenario_files, and MOCAT_config path
    # Edit this list to include the simulations you want to combine
    simulation_configs = [
        {
            'simulation_name': 'pmd_test_intensive',
            'scenario_files': ['bond_0k_5yr', 'bond_100k_5yr', 'bond_200k_5yr', 'bond_500k_5yr', 
                               'bond_750k_5yr', 'bond_1000k_5yr', 'bond_2000k_5yr'],
            'mocat_config_path': './OPUS/configuration/multi_single_species.json',
            'multi_species_names': ['S', 'Su', 'Sns']
        },
        {
            'simulation_name': 'pmd_test_extensive_5yr',
            'scenario_files': ['bond_0k_5yr', 'bond_100k_5yr', 'bond_200k_5yr', 'bond_500k_5yr', 
                               'bond_750k_5yr', 'bond_1000k_5yr', 'bond_2000k_5yr'],
            'mocat_config_path': './OPUS/configuration/multi_single_species.json',
            'multi_species_names': ['S', 'Su', 'Sns']
        }
    ]
    
    all_data = []
    
    # Load data from each simulation
    for config in simulation_configs:
        print(f"\nLoading data from {config['simulation_name']}...")
        
        # Load MOCAT config
        with open(config['mocat_config_path'], 'r') as f:
            mocat_config = json.load(f)
        
        # Create MOCAT instance (needed for PlotData)
        multi_species = MultiSpecies(config['multi_species_names'])
        MOCAT, _ = configure_mocat(mocat_config, multi_species=multi_species, grid_search=False)
        
        # Load simulation data
        bonds, nc, money, scenarios = load_simulation_data(
            config['simulation_name'],
            config['scenario_files'],
            MOCAT
        )
        
        if bonds:
            all_data.append({
                'simulation_name': config['simulation_name'],
                'bonds': bonds,
                'noncompliance': nc,
                'total_money': money,
                'scenarios': scenarios
            })
            print(f"  Loaded {len(bonds)} data points from {config['simulation_name']}")
        else:
            print(f"  No data loaded from {config['simulation_name']}")
    
    if not all_data:
        print("No data loaded! Check that simulation folders exist and contain results.")
        return
    
    # Create combined plot
    print("\nCreating combined plot...")
    create_combined_plot(all_data)
    
    # Save raw data to CSV
    print("\nSaving raw data to CSV...")
    df = save_raw_data_to_csv(all_data)
    
    print(f"\nSummary:")
    print(f"  Total data points: {len(df)}")
    print(f"  Simulations: {df['simulation_name'].unique()}")
    print(f"  Bond range: £{df['bond_amount'].min():,} - £{df['bond_amount'].max():,}")
    print(f"  Non-compliance range: {df['non_compliance_count'].min():.0f} - {df['non_compliance_count'].max():.0f}")


if __name__ == "__main__":
    main()

