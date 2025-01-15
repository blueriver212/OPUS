from utils.ScenarioNamer import generate_unique_name
from utils.ConstellationParameters import ConstellationParameters
from utils.EconParameters import EconParameters
from utils.MocatParameters import configure_mocat   
from utils.OpenAccessSolver import OpenAccessSolver
import pickle
import os
import json
import numpy as np
import time
import matplotlib.pyplot as plt

class IAMSolver:

    def __init__(self):
        self.output = None

    def iam_solver(self, stem, launch_pattern_type, parameter_file, n_workers, MOCAT_config):
        # Load mocat-pyssem model
        # There needs to be a way of defining which are your constellation satellites, and which are your fringe satellites.
        constellation_sats = "S"
        fringe_sats = "Su"

        MOCAT = configure_mocat(MOCAT_config, fringe_satellite=fringe_sats)
        print(MOCAT.scenario_properties.x0)

        # # # # This is just the x0 from MATLAB for testing purposes. Will be removed in final version
        data = [
            0, 4, 4, 64, 231, 28, 35, 41, 37, 2529, 208, 36, 14, 2, 14, 28, 113, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 283, 91, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 2, 2, 1, 1, 1, 5, 7, 2, 6, 11, 12, 6, 4, 3, 3, 3, 5, 0, 5, 0, 3, 0, 2, 15, 2, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 2, 1, 25, 16, 27, 34,
            46, 88, 120, 169, 97, 136, 232, 246, 290, 379, 537, 686, 909, 942, 999, 751, 562, 363, 396, 259, 229, 158, 164, 121, 78, 81, 185, 85, 60, 76, 78, 81, 102, 126, 95, 79,
            1, 1, 6, 19, 28, 121, 159, 233, 530, 334, 187, 187, 151, 77, 111, 53, 139, 91, 74, 31, 61, 63, 173, 18, 26, 53, 12, 6, 23, 4, 1, 6, 8, 11, 190, 173, 198, 59, 12, 9
        ]

        S = data[0:40]
        D = data[40:80]
        N = data[80:120]
        Su = data[120:160]

        MOCAT.scenario_properties.x0 = np.array([S, Su, N, D]).flatten()
        x0 = MOCAT.scenario_properties.x0

        # If testing using MOCAT x0 use:
        # x0 = MOCAT.scenario_properties.x0.T.values.flatten()

        sats_idx = MOCAT.scenario_properties.species_names.index(constellation_sats)
        fringe_idx = MOCAT.scenario_properties.species_names.index(fringe_sats)
        fringe_start_slice = (fringe_idx * MOCAT.scenario_properties.n_shells)
        fringe_end_slice = fringe_start_slice + MOCAT.scenario_properties.n_shells

        if parameter_file == 'benchmark':
            # Construct the file path
            file_path = os.path.join('scenarios', stem, parameter_file)
            
            # Create an empty file at the specified path
            with open(file_path, 'w') as f:
                pass  # `pass` creates an empty file
    
        # Update Parameters for scenario if supplied
        # skipping this for now - just working with the benchmark scenario

        # Build the cost function using the MOCAT model. 
        econ_params = EconParameters()
        econ_params.calculate_cost_fn_parameters(mocat=MOCAT) 

        # Initial Period Launch Rate
        constellation_params = ConstellationParameters('scenarios\parsets\constellation-parameters.csv')
        lam = constellation_params.define_initial_launch_rate(MOCAT, sats_idx, x0)
        # lam = constellation_params.fringe_sat_pop_feedback_controller()

        # Fringe population automomous controller. 
        launch_mask = np.ones((MOCAT.scenario_properties.n_shells,))

        species = MOCAT.scenario_properties.species_names
        n_shells = MOCAT.scenario_properties.n_shells

        if launch_pattern_type == "equilibrium":
            fringe = x0[fringe_start_slice:fringe_end_slice]
            solver_guess = 0.05 * np.array(fringe) #* launch_mask
            
            for idx in range(len(fringe)):
                lam[fringe_idx * n_shells + idx] = [solver_guess[idx]]

            count = 0
            for i in lam:
                if i is not None:
                    count += sum(i)  # Sum the elements inside the list

            print(count)

            # Solve for equilibrium launch rates
            open_access = OpenAccessSolver(MOCAT, solver_guess, launch_mask, x0, "linear", 
                                        econ_params, lam, n_workers, fringe_start_slice, fringe_end_slice)

            # This is now your estimate for the number of fringe satellites that should be launched.
            launch_rate = open_access.solver()

            for i in range(fringe_idx * MOCAT.scenario_properties.n_shells, (fringe_idx + 1) * MOCAT.scenario_properties.n_shells):
                lam[i] = [launch_rate[i - fringe_idx * MOCAT.scenario_properties.n_shells]]
            
            print(launch_rate)


        # Now we have the main iteration loop.
        model_horizon = MOCAT.scenario_properties.simulation_duration
        tf = np.arange(1, model_horizon + 1) 

        current_environment = x0 # Starts as initial population, and is in then updated. 
        dt = 5

        species_data = {sp: np.zeros((MOCAT.scenario_properties.simulation_duration, MOCAT.scenario_properties.n_shells)) for sp in species}

        for time_idx in tf:

            print("Starting year ", time_idx)
            
            # tspan = np.linspace(tf[time_idx], tf[time_idx + 1], time_step) # simulate for one year 
            tspan = np.linspace(0, 1, 2)
            
            # State of the environment during a simulation loop
            fringe_initial_guess = None

            # Propagate the model
            propagated_environment = MOCAT.propagate(tspan, current_environment, lam)
            propagated_environment = propagated_environment[-1, :]  # take the final list 
            
            # Update the constellation satellites for the next period - should only be 5%
            for i in range(sats_idx * MOCAT.scenario_properties.n_shells, (sats_idx + 1) * MOCAT.scenario_properties.n_shells):
                lam[i] = [propagated_environment[i] * (1 / dt)] 
            
            # Record propagated environment data
            for i, sp in enumerate(species):
                # 0 based index 
                species_data[sp][time_idx - 1] = propagated_environment[i * MOCAT.scenario_properties.n_shells:(i + 1) * MOCAT.scenario_properties.n_shells]

            if launch_pattern_type == "sat_feedback":
                continue  # skipping this for now.

            # Fringe Equilibrium Controller
            if launch_pattern_type == "equilibrium":
                start_time = time.time()
                print(f"Now starting period {time_idx}...")

                open_access = OpenAccessSolver(
                    MOCAT, fringe_initial_guess, launch_mask, propagated_environment, "linear",
                    econ_params, lam, n_workers, fringe_start_slice, fringe_end_slice)
                
                ror = open_access.fringe_rate_of_return(propagated_environment)
                collision_probability = open_access.calculate_probability_of_collision(propagated_environment)

                # Extract the relevant slice from lam
                lam_slice = lam[fringe_start_slice:fringe_end_slice]

                # Flatten the lam_slice to get a list of values
                lam_values = [item[0] for item in lam_slice]

                # Convert to numpy array for element-wise operations
                lam_values = np.array(lam_values)

                # Calculate solver_guess
                solver_guess = lam_values - lam_values * (ror - collision_probability) * launch_mask

                open_access = OpenAccessSolver(
                    MOCAT, solver_guess, launch_mask, propagated_environment, "linear",
                    econ_params, lam, n_workers, fringe_start_slice, fringe_end_slice)

                # Solve for equilibrium launch rates
                launch_rate = open_access.solver()

                # Update the initial conditions for the next period
                for i in range(fringe_idx * MOCAT.scenario_properties.n_shells, (fringe_idx + 1) * MOCAT.scenario_properties.n_shells):
                    lam[i] = [launch_rate[i - fringe_idx * MOCAT.scenario_properties.n_shells]]

                elapsed_time = time.time() - start_time
                print(f'Time taken for period {time_idx}: {elapsed_time:.2f} seconds')


            # Update the current environment
            current_environment = propagated_environment
        
        # Save the output
        self.output = species_data

        serializable_species_data = {sp: data.tolist() for sp, data in species_data.items()}

        # Save the serialized data to a JSON file
        with open('species_data.json', 'w') as json_file:
            json.dump(serializable_species_data, json_file, indent=4)

        print("species_data has been exported to species_data.json")

    
       
if __name__ == "__main__":
    # Behavior types
    launch_pattern_type_equilibrium="equilibrium"
    launch_pattern_type_feedback="sat_feedback"

    ####################
    ### 2. SCENARIO DEFINITIONS
    # Change to set scenarios and parallelization

    # Define the scenario to run. Store them in an array. Should be valid names of parameter set CSV files. 
    ## See examples in scenarios/parsets and compare to files named --parameters.csv for how to create new ones.
    scenario_files=[
                    "benchmark"
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
        iam_solver = IAMSolver()

        iam_solver.iam_solver(name, launch_pattern_type_equilibrium, scenario, n_workers, MOCAT_config)