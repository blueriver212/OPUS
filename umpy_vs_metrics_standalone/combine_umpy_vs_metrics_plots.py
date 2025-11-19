"""
Script to combine umpy_vs_metrics plots from Extensive and Intensive simulations.

This script:
1. Loads data from Extensive (pmd_test) and Intensive (pmd_test_intensive) simulation folders
2. Extracts raw data (Final UMPY, Collision Probability)
3. Calculates relative change in collision probability from Extensive £0 bond baseline
4. Creates a scatter plot showing:
   - Final UMPY vs Relative Change in Collision Probability (%)
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


def get_baseline_collision_probability(simulation_name, baseline_scenario, MOCAT):
    """
    Get the baseline collision probability from Extensive £0 bond scenario.
    
    Args:
        simulation_name: Name of the simulation (should be 'pmd_test' for Extensive)
        baseline_scenario: Name of the baseline scenario (e.g., 'bond_0k_5yr')
        MOCAT: MOCAT configuration object
    
    Returns:
        float: Baseline collision probability
    """
    simulation_folder = os.path.join("Results", simulation_name)
    scenario_folder = os.path.join(simulation_folder, baseline_scenario)
    
    if not os.path.exists(scenario_folder):
        raise ValueError(f"Baseline scenario folder not found: {scenario_folder}")
    
    try:
        plot_data = PlotData(baseline_scenario, scenario_folder, MOCAT)
        other_data = plot_data.get_other_data()
        
        if other_data is None:
            raise ValueError(f"No other_data found for baseline scenario {baseline_scenario}")
        
        # Get final timestep
        timesteps = sorted(other_data.keys(), key=int)
        if not timesteps:
            raise ValueError(f"No timesteps found for baseline scenario {baseline_scenario}")
        
        final_timestep = timesteps[-1]
        
        # Extract Final Collision Probability
        cp_data = other_data[final_timestep].get("collision_probability_all_species", [])
        if isinstance(cp_data, dict):
            baseline_cp = np.sum(list(cp_data.values()))
        else:
            baseline_cp = np.sum(cp_data) if cp_data is not None else 0
        
        return baseline_cp
    
    except Exception as e:
        raise ValueError(f"Error loading baseline collision probability: {e}")


def extract_bond_amount(scenario_name):
    """
    Extract bond amount from scenario name (e.g., 'bond_100k_5yr' -> 0.1 million).
    
    Args:
        scenario_name: Scenario name like 'bond_100k_5yr' or 'bond_0k_5yr'
    
    Returns:
        float: Bond amount in millions of dollars
    """
    # Pattern: bond_XXXk_YYyr where XXX is the amount in thousands
    match = re.search(r'bond_(\d+)k', scenario_name)
    if match:
        amount_thousands = int(match.group(1))
        return amount_thousands / 1000.0  # Convert to millions
    return 0.0


def load_simulation_data(simulation_name, scenario_files, MOCAT):
    """
    Load data from a simulation folder.
    
    Returns:
        tuple: (umpy_vals, collision_probs, scenario_names, bond_amounts)
    """
    simulation_folder = os.path.join("Results", simulation_name)
    
    if not os.path.exists(simulation_folder):
        print(f"Warning: {simulation_folder} does not exist. Skipping...")
        return [], [], [], []
    
    umpy_vals = []
    collision_probs = []
    scenario_names = []
    bond_amounts = []
    
    for scenario in scenario_files:
        scenario_folder = os.path.join(simulation_folder, scenario)
        
        if not os.path.exists(scenario_folder):
            print(f"Warning: {scenario_folder} does not exist. Skipping...")
            continue
        
        try:
            # Load PlotData
            plot_data = PlotData(scenario, scenario_folder, MOCAT)
            other_data = plot_data.get_other_data()
            species_data = plot_data.data
            
            if other_data is None:
                print(f"Warning: No other_data found for {scenario}. Skipping...")
                continue
            
            # Get final timestep
            timesteps = sorted(other_data.keys(), key=int)
            if not timesteps:
                print(f"Warning: No timesteps found for {scenario}. Skipping...")
                continue
            
            final_timestep = timesteps[-1]
            
            # Extract Final UMPY
            umpy_data = other_data[final_timestep].get("umpy", [])
            if isinstance(umpy_data, dict):
                final_umpy = np.sum(list(umpy_data.values()))
            else:
                final_umpy = np.sum(umpy_data) if umpy_data is not None else 0
            
            # Extract Final Collision Probability
            cp_data = other_data[final_timestep].get("collision_probability_all_species", [])
            if isinstance(cp_data, dict):
                final_cp = np.sum(list(cp_data.values()))
            else:
                final_cp = np.sum(cp_data) if cp_data is not None else 0
            
            bond_amount = extract_bond_amount(scenario)
            
            umpy_vals.append(final_umpy)
            collision_probs.append(final_cp)
            scenario_names.append(f"{simulation_name}/{scenario}")
            bond_amounts.append(bond_amount)
            
        except Exception as e:
            print(f"Error loading {scenario}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return umpy_vals, collision_probs, scenario_names, bond_amounts


def create_combined_plot(all_data, baseline_cp, output_path=None):
    """
    Create scatter plot showing UMPY vs relative change in collision probability.
    
    Args:
        all_data: List of dicts with simulation data
        baseline_cp: Baseline collision probability from Extensive £0 bond
        output_path: Path to save the plot
    """
    if output_path is None:
        output_path = "Results/combined_umpy_vs_metrics.png"

    name_mapping = {
        'pmd_test': 'Extensive',
        'pmd_test_intensive': 'Intensive'
    }

    # Calculate relative change in collision probability
    all_x = []
    all_y = []
    
    for data in all_data:
        umpy = data['umpy']
        collision_probs = data['collision_probs']
        
        # Calculate relative change as percentage: ((cp - baseline) / baseline) * 100
        relative_changes = [((cp - baseline_cp) / baseline_cp) * 100 if baseline_cp > 0 else 0 
                           for cp in collision_probs]
        
        all_x.extend(umpy)
        all_y.extend(relative_changes)
    
    # Calculate limits
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
    
    colour_map = {"Extensive": "red", "Intensive": "green"}
    markers = ["s", "o"]
    
    for i, data in enumerate(all_data):
        umpy = data['umpy']
        collision_probs = data['collision_probs']
        bond_amounts = data['bond_amounts']
        sim_name = data['simulation_name']
        
        display_name = name_mapping.get(sim_name, sim_name)
        colour = colour_map.get(display_name, "black")
        marker = markers[i % len(markers)]
        
        # Calculate relative change
        relative_changes = [((cp - baseline_cp) / baseline_cp) * 100 if baseline_cp > 0 else 0 
                           for cp in collision_probs]
        
        # Plot: UMPY vs Relative Change in Collision Probability
        ax.scatter(
            umpy,
            relative_changes,
            s=150,
            marker=marker,
            color=colour,
            alpha=0.85,
            edgecolors="k",
            linewidths=1.5,
            label=display_name,
            zorder=3,
        )
        
        # Add bond amount labels to each point
        for x, y, bond_millions in zip(umpy, relative_changes, bond_amounts):
            # Format bond amount: $0.0M, $0.1M, $2.0M, etc.
            if bond_millions == 0:
                label = "$0.0M"
            elif bond_millions == int(bond_millions):
                # Whole number: $1M, $2M, etc.
                label = f"${int(bond_millions)}.0M"
            else:
                # Decimal: $0.1M, $0.5M, $1.5M, etc.
                label = f"${bond_millions:.1f}M"
            
            # Place label directly above the point
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(0, 10),  # Offset: 0 horizontal (centered), 10 points above
                textcoords='offset points',
                ha='center',  # Center align horizontally
                fontsize=10,
                fontweight='bold',
                color=colour,
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
    ax.legend(title="Disposal Type", fontsize=14, frameon=True, loc="upper left")
    if ax.get_legend():
        ax.get_legend().get_title().set_fontweight("black")
        ax.get_legend().get_title().set_fontsize(16)
        for text in ax.get_legend().get_texts():
            text.set_fontweight("black")
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Combined scatter plot saved to {output_path}")
    plt.close()


def save_raw_data_to_csv(all_data, baseline_cp, output_path=None):
    """
    Save raw data to CSV for further analysis.
    
    Args:
        all_data: List of dicts with simulation data
        baseline_cp: Baseline collision probability from Extensive £0 bond
        output_path: Path to save CSV (default: Results/combined_umpy_vs_metrics_data.csv)
    """
    if output_path is None:
        output_path = "Results/combined_umpy_vs_metrics_data.csv"
    
    rows = []
    for data in all_data:
        for umpy, cp, scenario, bond_amount in zip(
            data['umpy'], 
            data['collision_probs'],
            data['scenarios'],
            data['bond_amounts']
        ):
            relative_change = ((cp - baseline_cp) / baseline_cp) * 100 if baseline_cp > 0 else 0
            rows.append({
                'simulation_name': data['simulation_name'],
                'scenario': scenario,
                'bond_amount_millions': bond_amount,
                'final_umpy': umpy,
                'collision_probability': cp,
                'baseline_collision_probability': baseline_cp,
                'relative_change_percent': relative_change
            })
    
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Raw data saved to {output_path}")
    return df


def main():
    """
    Main function to combine umpy_vs_metrics plots from Extensive and Intensive simulations.
    
    Edit the simulation_configs list to specify which simulations to combine.
    
    Usage:
        python combine_umpy_vs_metrics_plots.py
    
    The script will:
        1. Load baseline collision probability from Extensive £0 bond
        2. Load data from each simulation folder specified
        3. Extract Final UMPY and Collision Probability
        4. Calculate relative change in collision probability from baseline
        5. Create scatter plot showing Final UMPY vs Relative Change in Collision Probability
        6. Save raw data to CSV
    """
    import json
    
    # Configuration: List of simulations to combine
    # Each entry should have: simulation_name, scenario_files, and MOCAT_config path
    # Edit this list to include the simulations you want to combine
    simulation_configs = [
        {
            'simulation_name': 'pmd_test',
            'scenario_files': ['bond_0k_5yr', 'bond_100k_5yr', 'bond_200k_5yr', 'bond_500k_5yr', 
                               'bond_750k_5yr', 'bond_1000k_5yr', 'bond_2000k_5yr'],
            'mocat_config_path': './OPUS/configuration/multi_single_species.json',
            'multi_species_names': ['S', 'Su', 'Sns']
        },
        {
            'simulation_name': 'pmd_test_intensive',
            'scenario_files': ['bond_0k_5yr', 'bond_100k_5yr', 'bond_200k_5yr', 'bond_500k_5yr', 
                               'bond_750k_5yr', 'bond_1000k_5yr', 'bond_2000k_5yr'],
            'mocat_config_path': './OPUS/configuration/multi_single_species.json',
            'multi_species_names': ['S', 'Su', 'Sns']
        }
    ]
    
    # Baseline configuration: Extensive £0 bond
    baseline_config = {
        'simulation_name': 'pmd_test',
        'baseline_scenario': 'bond_0k_5yr',
        'mocat_config_path': './OPUS/configuration/multi_single_species.json',
        'multi_species_names': ['S', 'Su', 'Sns']
    }
    
    # Load MOCAT config for baseline
    print("Loading baseline collision probability from Extensive £0 bond...")
    with open(baseline_config['mocat_config_path'], 'r') as f:
        mocat_config = json.load(f)
    
    # Create MOCAT instance for baseline
    multi_species = MultiSpecies(baseline_config['multi_species_names'])
    MOCAT_baseline, _ = configure_mocat(mocat_config, multi_species=multi_species, grid_search=False)
    
    # Get baseline collision probability
    try:
        baseline_cp = get_baseline_collision_probability(
            baseline_config['simulation_name'],
            baseline_config['baseline_scenario'],
            MOCAT_baseline
        )
        print(f"  Baseline collision probability: {baseline_cp:.6f}")
    except Exception as e:
        print(f"Error loading baseline: {e}")
        return
    
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
        umpy, cp, scenarios, bond_amounts = load_simulation_data(
            config['simulation_name'],
            config['scenario_files'],
            MOCAT
        )
        
        if umpy:
            all_data.append({
                'simulation_name': config['simulation_name'],
                'umpy': umpy,
                'collision_probs': cp,
                'scenarios': scenarios,
                'bond_amounts': bond_amounts
            })
            print(f"  Loaded {len(umpy)} data points from {config['simulation_name']}")
        else:
            print(f"  No data loaded from {config['simulation_name']}")
    
    if not all_data:
        print("No data loaded! Check that simulation folders exist and contain results.")
        return
    
    # Create combined plot
    print("\nCreating combined plot...")
    create_combined_plot(all_data, baseline_cp)
    
    # Save raw data to CSV
    print("\nSaving raw data to CSV...")
    df = save_raw_data_to_csv(all_data, baseline_cp)
    
    print(f"\nSummary:")
    print(f"  Total data points: {len(df)}")
    print(f"  Simulations: {df['simulation_name'].unique()}")
    print(f"  Baseline collision probability: {baseline_cp:.6f}")
    print(f"  UMPY range: {df['final_umpy'].min():.2f} - {df['final_umpy'].max():.2f} kg/year")
    print(f"  Relative change range: {df['relative_change_percent'].min():.2f}% - {df['relative_change_percent'].max():.2f}%")


if __name__ == "__main__":
    main()

