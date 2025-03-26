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
- pip (or Anaconda)

### Install dependencies
```bash
git clone https://github.com/blueriver212/OPUS.git
cd OPUS
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install numpy pandas scipy matplotlib
```

## Running a Simulation

### Edit `OPUS/main.py`
Set your desired scenarios and simulation name:
```python
scenario_files = ["Baseline", "bond_100k", "tax_1"]
simulation_name = "demo_run"
```

### Run the main script
```bash
python OPUS/main.py
```

### Outputs
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

You can:
- Add new policies via config files (e.g. new bonds, disposal rules)
- Modify economics (discount rate, revenue function, costs)
- Extend debris model parameters (shell count, object properties)
- Create new plotting functions or export formats

## References
- Original OPUS paper: https://arxiv.org/abs/2309.10252
- PMD Bond Paper: [PDF in repo]
- MOCAT-pySSEM model: https://github.com/ARCLab-MIT/pyssem/

## License
MIT License

---

For contributions, bug reports, or feature requests, open an issue or pull request on [GitHub](https://github.com/blueriver212/OPUS).
