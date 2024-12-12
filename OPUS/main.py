from utils.ScenarioNamer import generate_unique_name
from utils.ConstellationParameters import ConstellationParameters
from utils.EconParameters import EconParameters
from utils.MocatParameters import configure_mocat   
from utils.OpenAccessSolver import OpenAccessSolver
import os
import json
import numpy as np
import time

def iam_solver(stem, model_type, launch_pattern_type, parameter_file, n_workers, MOCAT_config):
    # Load mocat-pyssem model
    MOCAT = configure_mocat(MOCAT_config)
    print(MOCAT.scenario_properties.x0)

    # There needs to be a way of defining which are your constellation satellites, and which are your fringe satellites.
    constellation_sats = "S"
    fringe_sats = "Su"
    sats_idx = MOCAT.scenario_properties.species_names.index(constellation_sats)
    fringe_idx = MOCAT.scenario_properties.species_names.index(fringe_sats)
    start_slice = (fringe_idx * MOCAT.scenario_properties.n_shells)
    end_slice = start_slice + MOCAT.scenario_properties.n_shells

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
    lam = constellation_params.define_initial_launch_rate(MOCAT, sats_idx)
    # lam = constellation_params.fringe_sat_pop_feedback_controller()

    # Fringe population automomous controller. 
    launch_mask = np.ones((MOCAT.scenario_properties.n_shells,))

    species = MOCAT.scenario_properties.species_names
    n_shells = MOCAT.scenario_properties.n_shells
    x0 = MOCAT.scenario_properties.x0.T.values.flatten()

    if launch_pattern_type == "equilibrium":
        Sui = x0[((fringe_idx - 1) * n_shells):fringe_idx * n_shells]
        solver_guess = 0.05 * Sui * launch_mask
        
        for idx in range(len(Sui)):
            lam[fringe_idx * n_shells + idx] = [solver_guess[idx]]

        # lam[:, 1] =  # Unslotted objects -- 5% feedback rule

        # tspan = np.linspace(0, 10, MOCAT.scenario_properties.steps)

        # This will return each of the species defined and their starting population. 
        # S, D, N, Su, T. Each is an array of n_shells and the T is the timesteps
        # Although does not actually seem to be used?
        # OUT = MOCAT4S(tspan, x0, lam, VAR)

        # Uncomment to check economic values (if these functions are implemented)
        # ror = fringeRateOfReturn("linear", econ_params, OUT, location_indices, launch_mask)
        # collision_probability = [calculateCollisionProbability_MOCAT4S(OUT, VAR, i) for i in range(VAR['N_shell'])]

        # Solve for equilibrium launch rates
        open_access = OpenAccessSolver(MOCAT, solver_guess, launch_mask, x0, "linear", 
                                       econ_params, lam, n_workers, fringe_idx)

        # This is now your estimate for the number of fringe satellites that should be launched.
        launch_rate = open_access.solver(solver_guess, launch_mask)
        for i in range(fringe_idx * MOCAT.scenario_properties.n_shells, (fringe_idx + 1) * MOCAT.scenario_properties.n_shells):
            lam[i] = [launch_rate[i - fringe_idx * MOCAT.scenario_properties.n_shells]]
        
        print(launch_rate)

    
    # Now we have the main iteration loop.
    model_horizon = 10
    tf = np.arange(1, model_horizon + 1) 

    # Initial Population
    x0 = MOCAT.scenario_properties.x0.T.values.flatten()
    current_environment = x0 # Starts as initial population, and is in then updated. 
    time_step = 1
    dt = 5
    
    for time_idx in tf:
        print("Starting year ", time_idx)
        
        # tspan = np.linspace(tf[time_idx], tf[time_idx + 1], time_step) # simulate for one year 
        tspan = np.linspace(0, 1, 2)
        
        # State of the environment during a simulation loop
        fringe_initial_guess = None

        # Propogate the model
        propagated_environment = MOCAT.propagate(tspan, current_environment, lam)
        propagated_environment = propagated_environment[-1, :] # take the final list 
        
        for i in range(sats_idx * MOCAT.scenario_properties.n_shells, (sats_idx + 1) * MOCAT.scenario_properties.n_shells):
            lam[i] = [propagated_environment[i] * (1 / dt)] 
        
        # Calculate the constellations/slotted satellite controller
        #lam = constellation_params.constellation_buildup(MOCAT, sats_idx)

        # Fringe Equilibrium Controller
        if launch_pattern_type == "equilibrium":

            start_time = time.time()
            print(f"Now starting period {time_idx + 1}...")

            open_access  = OpenAccessSolver(MOCAT, fringe_initial_guess, launch_mask, propagated_environment, "linear",
                                            econ_params, lam, n_workers, fringe_idx)
            
            open_access.find_initial_guess()

            # Solve for equilibrium launch rates

            # Update the inital conditions for the next period

            # Save the results of the launch period


            elapsed_time = time.time() - start_time
            print(f'Time taken for period {time_idx + 1}: {elapsed_time:.2f} seconds')










       
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
        iam_solver(name, model_type, launch_pattern_type_equilibrium, scenario, n_workers, MOCAT_config)