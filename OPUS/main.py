from utils.ConstellationParameters import ConstellationParameters
from utils.EconParameters import EconParameters
from utils.MocatParameters import configure_mocat   
from utils.OpenAccessSolver import OpenAccessSolver
from utils.PostProcessing import PostProcessing
from utils.PlotHandler import PlotHandler
from utils.PostMissionDisposal import evaluate_pmd
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
import time

class IAMSolver:

    def __init__(self):
        self.output = None
        self.MOCAT = None
        self.econ_params_json = None
        self.pmd_linked_species = None

    @staticmethod
    def get_species_position_indexes(MOCAT, constellation_sats, fringe_sats, pmd_linked_species):
        """
            The MOCAT model works on arrays that are the number of shells x number of species.
            Often throughout the model, we see the original list being spliced. 
            This function returns the start and end slice of the species in the array.

            Inputs:
                MOCAT: The MOCAT model
                constellation_sats: The name of the constellation satellites
                fringe_sats: The name of the fringe satellites
        """
        constellation_sats_idx = MOCAT.scenario_properties.species_names.index(constellation_sats)
        constellation_start_slice = (constellation_sats_idx * MOCAT.scenario_properties.n_shells)
        constellation_end_slice = constellation_start_slice + MOCAT.scenario_properties.n_shells

        fringe_idx = MOCAT.scenario_properties.species_names.index(fringe_sats)
        fringe_start_slice = (fringe_idx * MOCAT.scenario_properties.n_shells)
        fringe_end_slice = fringe_start_slice + MOCAT.scenario_properties.n_shells

        derelict_idx = MOCAT.scenario_properties.species_names.index(pmd_linked_species)
        derelict_start_slice = (derelict_idx * MOCAT.scenario_properties.n_shells)
        derelict_end_slice = derelict_start_slice + MOCAT.scenario_properties.n_shells

        return constellation_start_slice, constellation_end_slice, fringe_start_slice, fringe_end_slice, derelict_start_slice, derelict_end_slice

    def iam_solver(self, scenario_name, MOCAT_config, simulation_name):
        """
            The main function that runs the IAM solver.
        """
        # Define the species that are part of the constellation and fringe
        constellation_sats = "S"
        fringe_sats = "Su"

        #########################
        ### CONFIGURE MOCAT MODEL
        #########################
        if self.MOCAT is None:
            self.MOCAT, self.econ_params_json, self.pmd_linked_species = configure_mocat(MOCAT_config, fringe_satellite=fringe_sats)
            print(self.MOCAT.scenario_properties.x0)

        # If testing using MOCAT x0 use:
        x0 = self.MOCAT.scenario_properties.x0.T.values.flatten()
        constellation_start_slice, constellation_end_slice, fringe_start_slice, fringe_end_slice, derelict_start_slice, derelict_end_slice = self.get_species_position_indexes(self.MOCAT, constellation_sats, fringe_sats, self.pmd_linked_species)
    
        #################################
        ### CONFIGURE ECONOMIC PARAMETERS
        #################################
        econ_params = EconParameters(self.econ_params_json, mocat=self.MOCAT)
        if scenario_name != "Baseline":
            econ_params.modify_params_for_simulation(scenario_name)
        else: # needs to be a better way of doing this 
            econ_params.bond = None
            econ_params.tax = 0

        econ_params.calculate_cost_fn_parameters()

        ############################
        ### CONSTELLATION PARAMETERS
        ############################
        constellation_params = ConstellationParameters('./OPUS/configuration/constellation-parameters.csv')
        lam = constellation_params.define_initial_launch_rate(self.MOCAT, constellation_start_slice, constellation_end_slice, x0)

        # Fringe population automomous controller. 
        launch_mask = np.ones((self.MOCAT.scenario_properties.n_shells,))

        # Solver guess is 5% of the current fringe satellites. Update The launch file.
        solver_guess = 0.05 * np.array(x0[fringe_start_slice:fringe_end_slice]) * launch_mask
        lam[fringe_start_slice:fringe_end_slice] = solver_guess

        ############################
        ### SOLVE FOR THE FIRST YEAR
        ############################
        open_access = OpenAccessSolver(self.MOCAT, solver_guess, launch_mask, x0, "linear", 
                                    econ_params, lam, fringe_start_slice, fringe_end_slice,
                                    derelict_start_slice, derelict_end_slice)

        # This is now the first year estimate for the number of fringe satellites that should be launched.
        launch_rate, col_probability_all_species, umpy, excess_returns = open_access.solver()

        lam[fringe_start_slice:fringe_end_slice] = launch_rate
        
        
        ####################
        ### SIMULATION LOOP
        # For each year, take the previous state of the environment, 
        # then use the sovler to calculate the optimal launch rate. 
        ####################
        model_horizon = self.MOCAT.scenario_properties.simulation_duration
        tf = np.arange(1, model_horizon + 1) 

        current_environment = x0 # Starts as initial population, and is in then updated. 

        species_data = {sp: np.zeros((self.MOCAT.scenario_properties.simulation_duration, self.MOCAT.scenario_properties.n_shells)) for sp in self.MOCAT.scenario_properties.species_names}

        # Store the ror, collision probability and the launch rate 
        simulation_results = {}

        for time_idx in tf:

            print("Starting year ", time_idx)
            
            # tspan = np.linspace(tf[time_idx], tf[time_idx + 1], time_step) # simulate for one year 
            tspan = np.linspace(0, 1, 2)
            
            # State of the environment during a simulation loop
            fringe_initial_guess = None

            # Propagate the model and take the final state of the environment
            propagated_environment = self.MOCAT.propagate(tspan, current_environment, lam)
            propagated_environment = propagated_environment[-1, :] 
            propagated_environment = evaluate_pmd(propagated_environment, econ_params.comp_rate, self.MOCAT.scenario_properties.species['active'][1].deltat,
                                                    fringe_start_slice, fringe_end_slice, derelict_start_slice, derelict_end_slice, econ_params)

            # Update the constellation satellites for the next period - should only be 5%.
            for i in range(constellation_start_slice, constellation_end_slice):
                if lam[i] is not None:
                    lam[i] = lam[i] * 0.05

            #lam = constellation_params.constellation_launch_rate_for_next_period(lam, sats_idx, x0, MOCAT)
        
            # Record propagated environment data
            for i, sp in enumerate(self.MOCAT.scenario_properties.species_names):
                # 0 based index 
                species_data[sp][time_idx - 1] = propagated_environment[i * self.MOCAT.scenario_properties.n_shells:(i + 1) * self.MOCAT.scenario_properties.n_shells]

            # Fringe Equilibrium Controller
            start_time = time.time()
            print(f"Now starting period {time_idx}...")

            open_access = OpenAccessSolver(
                self.MOCAT, fringe_initial_guess, launch_mask, propagated_environment, "linear",
                econ_params, lam, fringe_start_slice, fringe_end_slice, derelict_start_slice, derelict_end_slice)
            
            collision_probability = open_access.calculate_probability_of_collision(propagated_environment)
            ror = open_access.fringe_rate_of_return(propagated_environment, collision_probability)

            # Calculate solver_guess
            solver_guess = lam[fringe_start_slice:fringe_end_slice] - lam[fringe_start_slice:fringe_end_slice] * (ror - collision_probability) * launch_mask

            open_access = OpenAccessSolver(
                self.MOCAT, solver_guess, launch_mask, propagated_environment, "linear",
                econ_params, lam, fringe_start_slice, fringe_end_slice, derelict_start_slice, derelict_end_slice)

            # Solve for equilibrium launch rates
            launch_rate, col_probability_all_species, umpy, excess_returns = open_access.solver()

            # Update the initial conditions for the next period
            lam[fringe_start_slice:fringe_end_slice] = launch_rate

            elapsed_time = time.time() - start_time
            print(f'Time taken for period {time_idx}: {elapsed_time:.2f} seconds')

            # Update the current environment
            current_environment = propagated_environment

            # Save the results that will be used for plotting later
            simulation_results[time_idx] = {
                "ror": ror,
                "collision_probability": collision_probability,
                "launch_rate" : launch_rate, 
                "collision_probability_all_species": col_probability_all_species,
                "umpy": umpy, 
                "excess_returns": excess_returns
            }
        
        PostProcessing(self.MOCAT, scenario_name, simulation_name, species_data, simulation_results, econ_params)

    def get_mocat(self):
        return self.MOCAT

def run_scenario(scenario_name, MOCAT_config, simulation_name):
    """
    Create a new IAMSolver instance for each scenario, run the simulation,
    and return the result from get_mocat().
    """
    solver = IAMSolver()
    solver.iam_solver(scenario_name, MOCAT_config, simulation_name)
    return solver.get_mocat()

def process_scenario(scenario_name, MOCAT_config, simulation_name):
    iam_solver = IAMSolver()
    iam_solver.iam_solver(scenario_name, MOCAT_config, simulation_name)
    return iam_solver.get_mocat()

if __name__ == "__main__":
    ####################
    ### 2. SCENARIO DEFINITIONS
    # Change to set scenarios and parallelization

    # Define the scenario to run. Store them in an array. Should be valid names of parameter set CSV files. 
    ## See examples in scenarios/parsets and compare to files named --parameters.csv for how to create new ones.
    scenario_files=[
                    "Baseline",
                    # "bond_0k_25yr",
                    "bond_100k",
                    # # "bond_200k",
                    # # "bond_300k",
                    # "bond_500k",
                    "bond_800k",
                    "bond_1600k",
                    # "bond_100k_25yr",
                    # # "bond_200k_25yr",
                    # # "bond_300k_25yr",
                    # "bond_500k_25yr",
                    # "bond_800k_25yr",
                    # "tax_1",
                    # "tax_2"
                ]
    
    MOCAT_config = json.load(open("./OPUS/configuration/three_species.json"))

    simulation_name = "New-Bonds"

    iam_solver = IAMSolver()

    # no parallel processing
    # for scenario_name in scenario_files:
    #     # in the original code - they seem to look at both the equilibrium and the feedback. not sure why. I am going to implement feedback first. 
    #     iam_solver.iam_solver(scenario_name, MOCAT_config, simulation_name)

    # Parallel Processing
    # PlotHandler(iam_solver.get_mocat(), scenario_files, simulation_name)
    with ThreadPoolExecutor() as executor:
        # Map process_scenario function over scenario_files
        results = list(executor.map(process_scenario, scenario_files, [MOCAT_config]*len(scenario_files), [simulation_name]*len(scenario_files)))

    # if you just want to plot the results - and not re- run the simulation. You just need to pass an instance of the MOCAT model that you created. 
    MOCAT,_, _ = configure_mocat(MOCAT_config, fringe_satellite="Su")
    PlotHandler(MOCAT, scenario_files, simulation_name, comparison=True)