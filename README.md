# OPUS (Orbital Propagators Unified with Economic Systems)

OPUS is an open-source integrated assessment model (IAM) focused on orbital debris and economics. It couples a physics-based space debris propagation model and collision assessment, with an economic behavior model of satellite operators. OPUS is designed to help engineers and space policy analysts evaluate how fiscal mechanisms can improve the long-term sustainability of Low Earth Orbit (LEO).

## Purpose and Overview

The OPUS model was originaly developed under a [NASA Roses Grant](https://arxiv.org/abs/2309.10252) to study market-based policies for orbital debris mitigation. The initial verision was in [Matlab](https://github.com/akhilrao/OPUS). The original model uses MOCAT-4S and the GMPHD filter as the debris component, this repository has forked the modernised MOCAT-pySSEM debris model and also forking the original code.

**Key purpose:** OPUS simulates how economic incentives affect operator behavior and, in turn, the space debris environment. For example, a **PMD bond** requires an operator to deposit a set amount per satellite, refundable upon proper disposal. Failure to deorbit forfeits the bond, which could fund debris removal. Similarly, **orbital-use fees** act as a tax on satellites in orbit. By modeling these mechanisms, OPUS helps quantify outcomes like:

- **Compliance rates**: What fraction of satellite missions comply under different PMD deorbit rules?
- **Launch behavior**: How many satellites do operators choose to launch when faced with bonds or fees?
- **Debris population**: How the number of derelict objects and collision risk evolve over decades.
- **Sustainability metrics**: Long-term indicators such as *Undisposed Mass Per Year (UMPY)*, which measures the mass of uncontrolled objects added to orbit annually.

Ultimately, OPUS is a tool to **compare policy scenarios** and inform data-driven decisions for space environmental management.

## Model Features and Components

OPUS consists of two main components:

- **Economic Module**: Models how satellite operators make decisions under open-access market conditions. It calculates launch rates, PMD compliance, and profit-maximizing behavior in response to policy interventions like bonds or taxes.

- **Debris Module (MOCAT-pySSEM)**: A physics-based orbital debris model that simulates LEO as discrete altitude shells and models population changes due to launch, atmospheric drag, collisions, and PMD compliance.

### Other Key Features:
- Multiple species: Active satellites, debris, rocket bodies
- Configurable policy scenarios: PMD bonds, orbital-use fees, disposal guidelines
- Extensible configuration system with CSV and JSON files
- Results exported as JSON with automated plotting tools

## Installation and Setup

### Requirements
- Python 3.9+

### Install dependencies
```bash
git clone https://github.com/blueriver212/OPUS.git
cd OPUS
pip install -r requirements.txt
```

## Running a Simulation

### Simulation configuration. 
A OPUS model uses the same configuration as [MOCAT-pySSEM](https://github.com/ARCLab-MIT/pyssem/). There is an example in ```./OPUS/configuration/three_species.json```.

For the baseline scenario, the economic parameters can be configured similar to the example below.

```json
    "OPUS" :{
        "sat_lifetime" : 5, // Satellite Active Lifetime (e.g 5 years)
        "disposal_time" : 5, // PMD Rule (e.g. 5 years)
        "discount_rate" : 0.05, // Annual discount rate
        "intercept" : 7.5e5, // Annual revenue per year, per orbital shell
        "coef" : 1.0e2, 
        "tax": 0, // Tax scaled with collision risk. 
        "delta_v_cost": 1000, // Station Keeping Cost
        "lift_price": 5000, // $/kg price
        "prob_of_non_compliance": 0 // Non compliance rate. If using bonds, this will be changed later anyway
    },
```

The baseline will be automatically applied to each of the scenarios. To then create individual scenarios, a csv must be created with the parameters to change. The name of the csv is the simulation name. As an example, to create a bond of $800k and a PMD rule of 25 years. The name of the file will be used on main.py.

```csv
parameter_type,parameter_name,parameter_value
econ,bond,500000
econ,disposal_time,25
```

### Edit `OPUS/main.py`
Set your desired scenarios and simulation name, anything other than baseline must be a csv in the configration folder:
```python
scenario_files = ["Baseline", "bond_800k"]
simulation_name = "demo_run"
```

### Run the main script
```bash
python OPUS/main.py
```

### OutputsÆ
Results are stored in `Results/<simulation_name>/<scenario_name>/`, including:
- Time-series data (species populations)
- Economic metrics (ROR, compliance, collision risk)

## Reproducing the Paper Results

To replicate results from the paper *"Post-Mission Disposal Bond for the Long-Term Sustainability of Space":*

1. Set `scenario_files` to include:
   - `Baseline`, `bond_0k_25yr`
   - `bond_100k`, `bond_100k_25yr`, ..., `bond_800k`, `bond_800k_25yr`
   - `tax_1`, `tax_2`

2. Run the simulation:
```bash
python OPUS/main.py
```

3. Generate comparison plots:
```python
from OPUS.utils.PlotHandler import PlotHandler
from OPUS.utils.MocatParameters import configure_mocat
import json
MOCAT_config = json.load(open("./OPUS/configuration/three_species.json"))
MOCAT_model, _, _ = configure_mocat(MOCAT_config, fringe_satellite="Su")
PlotHandler(MOCAT_model, scenario_files, simulation_name, comparison=True)
```

## Customization and Extension
There are various ways to complete policy and configuration comparison, including:
- Add new policies via config files (e.g. new bonds, disposal rules)
- Modify economics (discount rate, revenue function, costs)
- Extend debris model parameters (shell count, object properties)
- Create new plotting functions or export formats

## References
- Original OPUS paper: https://arxiv.org/abs/2309.10252
- PMD Bond Paper: # Will be uploaded after conference.
- MOCAT-pySSEM model: https://github.com/ARCLab-MIT/pyssem/

## License
MIT License
---
For contributions, bug reports, or feature requests, open an issue or pull request on [GitHub](https://github.com/blueriver212/OPUS) or contact Indigo at: indigo.brownhall.20@ucl.ac.uk
