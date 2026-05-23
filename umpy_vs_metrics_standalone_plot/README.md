# UMPY vs Metrics Standalone Plots

This folder contains standalone scripts to create plots from CSV data.

## Contents

### Plot 1: Welfare and UMPY % Change

- `total_satellites_and_umpy_percent_change_data.csv` - Raw data containing:
  - Bond amounts (in k and millions)
  - PMD lifetime (5yr or 25yr)
  - Total satellites count
  - Welfare % change
  - UMPY % change

- `plot_from_csv.py` - Python script that reads the CSV and creates the plot

- `total_satellites_and_umpy_percent_change.png` - Generated plot (created when you run the script)

### Plot 2: Object Counts vs Bond by Species with UMPY Relative

- `object_counts_vs_bond_by_species_with_umpy_relative_data.csv` - Raw data containing:
  - Bond amounts (in k and millions)
  - PMD lifetime (5yr or 25yr)
  - Species (S, Su, Sns)
  - Active satellites count
  - Compliant derelicts count
  - Non-compliant derelicts count
  - UMPY % change from baseline

- `plot_object_counts_by_species.py` - Python script that reads the CSV and creates the 2x2 subplot

- `object_counts_vs_bond_by_species_with_umpy_relative.png` - Generated plot (created when you run the script)

### Plot 3: UMPY vs Collision Probability

- `combined_umpy_vs_metrics_data.csv` - Raw data containing:
  - Final UMPY values
  - Collision probabilities
  - Relative change in collision probability (%)
  - Bond amounts
  - Simulation names and scenarios

- `plot_umpy_vs_collision_probability.py` - Python script that reads the CSV and creates the plot

- `combined_umpy_vs_metrics.png` - Generated plot (created when you run the script)

## Usage

1. Make sure you have the required dependencies:
   ```bash
   pip install pandas matplotlib numpy
   ```

2. Run the plotting scripts:
   ```bash
   # For Welfare and UMPY % Change plot (main plot)
   python plot_from_csv.py
   
   # For Object Counts vs Bond by Species with UMPY Relative (2x2 subplot)
   python plot_object_counts_by_species.py
   
   # For UMPY vs Collision Probability plot
   python plot_umpy_vs_collision_probability.py
   ```

## Plot 1 Details: Welfare and UMPY % Change

The plot shows:
- **X-axis**: Lifetime Bond Amount ($ millions)
- **Y-axis**: Relative Change (%)
- **Lines**: 
  - Welfare % Change (5yr and 25yr) - blue lines
  - UMPY % Change (5yr and 25yr) - red lines
- **Legend**: Shows all four lines with their PMD lifetimes

### Data Format for Plot 1

The CSV file contains the following columns:
- `bond_amount_k`: Bond amount in thousands
- `bond_amount_millions`: Bond amount in millions of dollars
- `pmd_lifetime_years`: PMD lifetime (5 or 25 years)
- `total_satellites`: Total number of satellites (may be None for UMPY-only rows)
- `satellites_percent_change`: Welfare % change from baseline (may be None for UMPY-only rows)
- `umpy`: UMPY value (currently None, reserved for future use)
- `umpy_percent_change`: UMPY % change from baseline (may be None for satellite-only rows)

## Plot 2 Details: Object Counts vs Bond by Species with UMPY Relative

The plot shows a 2x2 subplot:
- **Top row**: Three plots (S, Su, Sns) showing:
  - Active satellites (blue, solid for 5yr, dashed for 25yr)
  - Compliant derelicts (green, solid for 5yr, dashed for 25yr)
  - Non-compliant derelicts (red, solid for 5yr, dashed for 25yr)
- **Bottom right**: Relative change in final year UMPY from baseline (%) vs bond amount
- **X-axis**: Lifetime Bond Amount ($ millions)
- **Y-axis**: Number of Objects (for species plots) or Relative Change in UMPY (%) (for UMPY plot)

### Data Format for Plot 2

The CSV file contains the following columns:
- `bond_amount_k`: Bond amount in thousands
- `bond_amount_millions`: Bond amount in millions of dollars
- `pmd_lifetime_years`: PMD lifetime (5 or 25 years)
- `species`: Species name (S, Su, or Sns)
- `active_satellites`: Number of active satellites (may be None for UMPY-only rows)
- `compliant_derelicts`: Number of compliant derelicts (may be None for UMPY-only rows)
- `non_compliant_derelicts`: Number of non-compliant derelicts (may be None for UMPY-only rows)
- `umpy_percent_change`: UMPY % change from baseline (may be None for species-only rows)

## Plot 3 Details: UMPY vs Collision Probability

The plot shows:
- **X-axis**: Final UMPY (kg/year)
- **Y-axis**: Relative Change in Collision Probability (%) from baseline
- **Legend**: "Bond Refunded at" - showing different disposal types
- **Labels**: Each point is labeled with the bond amount (e.g., "$0.1M", "$2.0M")

### Data Format for Plot 2

The CSV file contains the following columns:
- `simulation_name`: Name of the simulation (e.g., 'extensive_new', 'intensive')
- `scenario`: Scenario name (e.g., 'bond_100k_5yr')
- `bond_amount_millions`: Bond amount in millions of dollars
- `final_umpy`: Final UMPY value (kg/year)
- `collision_probability`: Raw collision probability
- `baseline_collision_probability`: Baseline collision probability used for comparison
- `relative_change_percent`: Relative change in collision probability as a percentage

