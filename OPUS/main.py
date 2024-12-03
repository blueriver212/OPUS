from utils.ScenarioNamer import generate_unique_name
from utils.ConstellationParameters import ConstellationParameters
from utils.EconParameters import EconParameters
from utils.MocatParameters import configure_mocat   
import os
from pyssem.model import Model
import json
import numpy as np

def iam_solver(stem, model_type, launch_pattern_type, parameter_file, n_workers, MOCAT_config):

    # Load mocat-pyssem model
    MOCAT = configure_mocat(MOCAT_config)
    print(MOCAT.scenario_properties.x0)
 
    # Load the Constellations
    constellation_params = ConstellationParameters('scenarios\parsets\constellation-parameters.csv')

    # Economic Parameters
    econ_params = EconParameters()

    if parameter_file == 'benchmark':
        # Construct the file path
        file_path = os.path.join('scenarios', stem, parameter_file)
        
        # Create an empty file at the specified path
        with open(file_path, 'w') as f:
            pass  # `pass` creates an empty file

    
    # Update Parameters for scenario if supplied
    # skipping this for now - just working with the benchmark scenario

    # Build the cost function using the MOCAT model. 
    econ_params.calculate_cost_fn_parameters(mocat=MOCAT) 

    # Initial Period Launch Rate
    lam = constellation_params.define_initial_launch_rate(MOCAT)

    lam = constellation_params.fringe_sat_pop_feedback_controller()


    # Fringe population automomous controller. 
    # TODO profiles need to e created and tested. Code block describes intent for future development.
    Sui=None
    launch_mask = np.ones((MOCAT.scenario_properties.n_shells, 1))
    if launch_pattern_type == "equilibrium":
        solver_guess = 0.05 * Sui * launch_mask
        lam[:, 1] = solver_guess  # Unslotted objects -- 5% feedback rule

        tspan = np.linspace(0, 10, MOCAT.scenario_properties.steps)

        # This will return each of the species defined and their starting population. 
        OUT = MOCAT4S(tspan, x0, lam, VAR)

        # Uncomment to check economic values (if these functions are implemented)
        # ror = fringeRateOfReturn("linear", econ_params, OUT, location_indices, launch_mask)
        # collision_probability = [calculateCollisionProbability_MOCAT4S(OUT, VAR, i) for i in range(VAR['N_shell'])]

        # Solve for equilibrium launch rates
        lam[:, 1] = openAccessSolver(solver_guess, launch_mask, "MOCAT", VAR, x0, "linear", econ_params, lam[:, 0], n_workers)
       
if __name__ == "__main__":
    # Behavior types
    launch_pattern_type_equilibrium="equilibrium"
    launch_pattern_type_feedback="sat_feedback"

    ####################
    ### 2. SCENARIO DEFINITIONS
    # Change to set scenarios and parallelization

    # Choose propagator
    model_type="MOCAT" # Either "MOCAT" or "GMPHD"

    # Define the scenario to run. Store them in an array. Should be valid names of parameter set CSV files. 
    ## See examples in scenarios/parsets and compare to files named --parameters.csv for how to create new ones.
    scenario_files=[
                    "benchmark",
                    "scenarios/parsets/tax_1.csv",
                    "scenarios/parsets/tax_2.csv",
                    ]

    # Define the length of the horizon over which to solve (years)
    # You now need to set it using the mocat.json, to ensure that it is consistent with the scenario

    # Set number of workers for parallelization, set to 0 for maximum
    n_workers=0

    # Use this when not testing
    # name = generate_unique_name()
    name = 'recycling-whimsical-pays-Fig'

    # Check to see if a directory in scnenarios exists, else create it
    if not os.path.exists(f"scenarios/{name}"):
        os.makedirs(f"scenarios/{name}")
    else:
        print("Directory already exists")

    
    print(os.getcwd())

    MOCAT_config = json.load(open("scenarios/parsets/three_species.json"))

    for scenario in scenario_files:

        # in the original code - they seem to look at both the equilibrium and the feedback. not sure why. I am going to implement feedback first. 
        iam_solver(name, model_type, launch_pattern_type_feedback, scenario, n_workers, MOCAT_config)