# OPUS: Orbital Propagators Unified with Economic Systems

**Version:** 1.0  
**DOI:** [To be assigned upon publication]  
**License:** MIT License (see LICENSE file)

## Overview

OPUS is an open-source integrated assessment model (IAM) that couples a physics-based orbital debris propagation model (MOCAT-pySSEM) with an economic behavior model of satellite operators. The model simulates how fiscal mechanisms, such as post-mission disposal (PMD) bonds and orbital-use fees, affect operator behavior and the long-term sustainability of Low Earth Orbit (LEO).

### Key Capabilities

- **Economic Module**: Models satellite operator decision-making under open-access market conditions, calculating launch rates, PMD compliance, and profit-maximizing behavior in response to policy interventions
- **Debris Module**: Physics-based orbital debris model (MOCAT-pySSEM) that simulates LEO as discrete altitude shells, modeling population changes due to launch, atmospheric drag, collisions, and PMD compliance
- **Policy Scenarios**: Configurable PMD bonds, orbital-use fees, and disposal guidelines
- **Multi-Species Support**: Handles active satellites, uncontrolled satellites, small satellites, debris, and rocket bodies

## Installation

### System Requirements

- **Operating System**: macOS, Linux, or Windows
- **Python Version**: Python 3.9 or higher
- **Memory**: Minimum 8 GB RAM (16 GB recommended for large simulations)
- **Storage**: ~500 MB for installation, additional space for results (varies by simulation size)

### Dependencies

The model requires the following Python packages (see `requirements.txt`):

- `numpy>=1.2`
- `pandas>=2.0`
- `scipy>=1.10`
- `sympy>=1.11`
- `matplotlib`
- `pyssem==0.1.dev294` (MOCAT-pySSEM debris model)
- `joblib`
- `loky`
- Additional packages: `tqdm`, `gdown`, `requests`, `bs4`, `imageio`, `sphinx_rtd_theme`

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/blueriver212/OPUS.git
   cd OPUS
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Typical installation time:** 5-10 minutes on a modern computer with a stable internet connection.

3. **Verify installation:**
   ```bash
   python -c "import numpy, pandas, scipy, pyssem; print('Installation successful')"
   ```

## Running the Model

### Basic Workflow

The OPUS model is executed through `OPUS/main.py`. The simulation requires:
1. A MOCAT configuration JSON file (defines species, orbital parameters, and economic baseline)
2. Bond scenario CSV files (optional, for policy scenarios)
3. Configuration of simulation parameters in `main.py`

### Running a Normal Bond Behavior Simulation

This section describes how to run simulations with standard bond behavior (baseline economic parameters).

#### Step 1: Configure the Simulation

Open `OPUS/main.py` and modify the following parameters in the `if __name__ == "__main__":` block (around line 430):

```python
# Set bond amounts and disposal lifetimes to test
bond_amounts = [700000, 1000000, 2000000]  # Bond amounts in USD
lifetimes = [25]  # Disposal time requirement in years

# Set baseline flag (True to include baseline scenario, False to skip)
baseline = False

# Select the MOCAT configuration file (normal bond behavior)
MOCAT_config = json.load(open("./OPUS/configuration/multi_single_species.json"))

# Set simulation name (creates output directory)
simulation_name = "my_simulation"

# Define species names (must match sym_name in MOCAT config)
multi_species_names = ["S", "Su", "Sns"]  # S=active satellites, Su=uncontrolled, Sns=small satellites

# Define which species are subject to bonds for market competition mapping (empty list = no market competition)
bonded_species_names = []
```

#### Step 2: Run the Simulation

Execute the main script:

```bash
python OPUS/main.py
```

The script will:
- Automatically create bond configuration CSV files in `OPUS/configuration/` if they don't exist
- Run simulations for each bond scenario in parallel (using `ProcessPoolExecutor`)
- Save results to `Results/<simulation_name>/<scenario_name>/`

#### Step 3: View Results

Results are stored in JSON format in `Results/<simulation_name>/<scenario_name>/`, including:
- Time-series data (species populations by altitude shell)
- Economic metrics (rate of return, compliance rates, collision probabilities)
- Policy outcomes (bond revenue, welfare, tax revenue)

Plots are automatically generated and saved in the same directory.

**Typical run time:** 
- Single scenario (20-year simulation): 5-15 minutes
- Multiple scenarios (parallel processing): Scales with number of CPU cores

### Running Simulations with Competitors

To simulate competitive market conditions with different operator characteristics, use the `multi_single_species_joey.json` configuration file. This configuration includes:
- Higher profit margins (`Pm` values: 0.95 for S, 0.65 for Su and Sns vs. 0.45 baseline)
- Different PMD behavior (controlled reentries only, higher controlled PMD rates)

#### Step 1: Modify Configuration in `main.py`

Change the MOCAT configuration file path:

```python
# Use competitor configuration instead of normal configuration
MOCAT_config = json.load(open("./OPUS/configuration/multi_single_species_joey.json"))
```

All other parameters remain the same as the normal bond behavior simulation.

#### Step 2: Run the Simulation

Execute the script as before:

```bash
python OPUS/main.py
```

The simulation will run with competitor economic parameters, allowing comparison of how different operator characteristics affect policy outcomes.

### Example: Complete Workflow

Here is a complete example demonstrating both scenarios:

```python
# In OPUS/main.py

if __name__ == "__main__":
    # Configuration for bond scenarios
    bond_amounts = [0, 100000, 700000, 1000000, 2000000]
    lifetimes = [25]
    baseline = False
    
    # Ensure bond configuration files exist
    bond_scenario_names = ensure_bond_config_files(bond_amounts, lifetimes)
    scenario_files = []
    if baseline:
        scenario_files.append("Baseline")
    scenario_files.extend(bond_scenario_names)
    
    # NORMAL BOND BEHAVIOR: Use multi_single_species.json
    MOCAT_config = json.load(open("./OPUS/configuration/multi_single_species.json"))
    
    # COMPETITOR SCENARIO: Uncomment to use competitor configuration
    # MOCAT_config = json.load(open("./OPUS/configuration/multi_single_species_joey.json"))
    
    simulation_name = "example_run"
    if not os.path.exists(f"./Results/{simulation_name}"):
        os.makedirs(f"./Results/{simulation_name}")
    
    multi_species_names = ["S", "Su", "Sns"]
    bonded_species_names = []
    
    # Run simulations (parallel processing)
    with ProcessPoolExecutor() as executor:
        list(executor.map(process_scenario, 
                         scenario_files, 
                         [MOCAT_config] * len(scenario_files), 
                         [simulation_name] * len(scenario_files), 
                         [multi_species_names] * len(scenario_files),
                         [bonded_species_names] * len(scenario_files)))
    
    # Generate comparison plots
    multi_species = MultiSpecies(multi_species_names)
    MOCAT, _ = configure_mocat(MOCAT_config, multi_species=multi_species, grid_search=False)
    PlotHandler(MOCAT, scenario_files, simulation_name, comparison=True)
```

## Configuration Files

### MOCAT Configuration JSON

The MOCAT configuration file (`multi_single_species.json` or `multi_single_species_joey.json`) defines:
- **Simulation properties**: Start date, duration, altitude range, number of shells
- **Species definitions**: Physical properties (mass, radius, area), orbital parameters, economic parameters
- **Debris model settings**: Density model, collision parameters, fragmentation settings

Key differences between configurations:
- **`multi_single_species.json`**: Baseline economic parameters (`Pm=0.45` for all active species)
- **`multi_single_species_joey.json`**: Competitor scenario (`Pm=0.95` for S, `Pm=0.65` for Su/Sns)

### Bond Scenario CSV Files

Bond scenarios are defined in CSV files located in `OPUS/configuration/`. The naming convention is:
```
bond_<amount>k_<lifetime>yr.csv
```

Example (`bond_700k_25yr.csv`):
```csv
parameter_type,parameter_name,parameter_value
econ,bond,700000
econ,disposal_time,25
```

These files override the baseline economic parameters from the MOCAT configuration JSON.

## Output and Results

### Output Structure

```
Results/
└── <simulation_name>/
    ├── <scenario_name>/
    │   ├── species_data.json          # Time-series population data
    │   ├── simulation_results.json    # Economic and policy metrics
    │   ├── econ_params.json           # Economic parameters used
    │   └── plots/                     # Generated visualizations
    └── comparisons/                   # Comparison plots across scenarios
```

### Key Output Metrics

- **Species populations**: Number of objects by species and altitude shell over time
- **Launch rates**: Annual launch rates by species
- **Compliance rates**: Fraction of satellites complying with PMD requirements
- **Collision probabilities**: Per-species collision risk
- **Economic metrics**: Rate of return, welfare, bond revenue, tax revenue
- **UMPY**: Undisposed Mass Per Year (sustainability metric)

## Reproducing Published Results

To reproduce results from the associated publication:

1. **Set bond scenarios:**
   ```python
   bond_amounts = [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1200000, 1300000, 1400000, 1500000, 2000000]
   lifetimes = [25]
   baseline = True  # Include baseline scenario
   ```

2. **Run simulation:**
   ```bash
   python OPUS/main.py
   ```

3. **Generate comparison plots:**
   The script automatically generates comparison plots. Alternatively, use:
   ```python
   from OPUS.utils.PlotHandler import PlotHandler
   from OPUS.utils.MocatParameters import configure_mocat
   from OPUS.utils.MultiSpecies import MultiSpecies
   import json
   
   MOCAT_config = json.load(open("./OPUS/configuration/multi_single_species.json"))
   multi_species_names = ["S", "Su", "Sns"]
   multi_species = MultiSpecies(multi_species_names)
   MOCAT, _ = configure_mocat(MOCAT_config, multi_species=multi_species, grid_search=False)
   PlotHandler(MOCAT, scenario_files, simulation_name, comparison=True)
   ```

## Test Dataset

The model includes example configurations and automatically generates test scenarios. A minimal test run can be executed with:

```python
# In main.py, set:
bond_amounts = [700000]
lifetimes = [25]
baseline = False
simulation_name = "test_run"
```

This will run a single scenario and generate all outputs, suitable for verifying installation and basic functionality.

**Expected test run time:** 1-4 hours depending on configuration

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`
2. **File not found errors**: Verify you are running from the repository root directory
3. **Memory errors**: Reduce simulation duration or number of shells in the MOCAT configuration
4. **Parallel processing errors**: Set `parallel_processing: false` in the MOCAT configuration JSON

### Getting Help

- **Issues**: Report bugs or questions via GitHub Issues
- **Contact**: indigo.brownhall.20@ucl.ac.uk

## Code Availability

This code is available as:
- **Supplementary Software**: Included with the publication
- **Repository**: https://github.com/blueriver212/OPUS
- **DOI**: [To be assigned upon publication]

### License

This code is released under the MIT License. See LICENSE file for details.

### Restrictions

No restrictions on code availability. The code is freely accessible and can be used, modified, and distributed under the terms of the MIT License.

## Citation

If you use OPUS in your research, please cite:

```
[Citation information to be added upon publication]
```

## References

- Original OPUS paper: https://arxiv.org/abs/2309.10252
- MOCAT-pySSEM model: https://github.com/ARCLab-MIT/pyssem/
- PMD Bond Paper: [To be added upon publication]

## Acknowledgments

 The model builds upon the MOCAT-pySSEM debris propagation framework developed by Unversity College London (UCL) and MIT Astrodynamics, Space Robotics, and Controls Laboratory (ARCLab).

---

**Last Updated:** February 2026  
**Maintained by:** Indigo Brownhall (indigo.brownhall.20@ucl.ac.uk)
