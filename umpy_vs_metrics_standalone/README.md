# UMPY vs Metrics Plotting Script

This standalone package generates scatter plots comparing Final UMPY vs Relative Change in Collision Probability for Extensive and Intensive simulations.

## Requirements

- Python 3.8+
- pyssem library (must be installed separately)
- Dependencies listed in `requirements.txt`

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure `pyssem` is installed and accessible. The script requires access to the `pyssem` library.

## Usage

1. Edit the `main()` function in `combine_umpy_vs_metrics_plots.py` to configure:
   - Simulation names and scenario files
   - MOCAT configuration file paths
   - Multi-species names

2. Ensure your simulation results are in the `Results/` folder with the structure:
   ```
   Results/
     <simulation_name>/
       <scenario_name>/
         species_data_<scenario_name>.json
         other_results_<scenario_name>.json
         econ_params_<scenario_name>.json
   ```

3. Run the script:
```bash
python combine_umpy_vs_metrics_plots.py
```

## Output

The script generates:
- `Results/combined_umpy_vs_metrics.png` - Scatter plot showing Final UMPY vs Relative Change in Collision Probability (%)
- `Results/combined_umpy_vs_metrics_data.csv` - Raw data in CSV format

## Configuration

Edit the `simulation_configs` and `baseline_config` dictionaries in the `main()` function to customize:
- Simulation names (e.g., 'pmd_test', 'pmd_test_intensive')
- Scenario files (e.g., bond amounts and lifetimes)
- MOCAT configuration file paths
- Multi-species names

## Dependencies

- **PlotHandler.py**: Contains the `PlotData` class for loading simulation data
- **MultiSpecies.py**: Handles multi-species configuration
- **MocatParameters.py**: Configures MOCAT model instances
- **EconParameters.py**: Economic parameter calculations (used by MocatParameters)

## Notes

- The script calculates relative change in collision probability relative to the Extensive £0 bond baseline
- Bond amounts are automatically extracted from scenario names (e.g., 'bond_100k_5yr' → $0.1M)
- Labels are placed directly above each scatter point showing the bond amount

