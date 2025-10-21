from utils.ConstellationParameters import ConstellationParameters
from utils.EconParameters import EconParameters
from utils.MocatParameters import configure_mocat   
from utils.OpenAccessSolver import OpenAccessSolver
from utils.PostProcessing import PostProcessing
from utils.PlotHandler import PlotHandler
from utils.PostMissionDisposal import evaluate_pmd, evaluate_pmd_elliptical
from utils.MultiSpecies import MultiSpecies
from utils.MultiSpeciesOpenAccessSolver import MultiSpeciesOpenAccessSolver
from utils.Helpers import insert_launches_into_lam
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from datetime import timedelta
import time

class IAMSolver:

    def __init__(self):
        """
            Initialize the IAMSolver class.
            This class is responsible for running the IAM solver and managing the MOCAT model.
        """
        self.output = None
        self.MOCAT = None
        self.econ_params_json = None
        self.pmd_linked_species = None

    @staticmethod
    def get_species_position_indexes(MOCAT, constellation_sats):
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

        return constellation_start_slice, constellation_end_slice

    def _apply_replacement_floor(self, solver_guess_array, environment_array, multi_species):
        """
            Ensure each species starts the solve with at least the replacement launches
            implied by the active population removed via PMD.
        """
        replacement_caps = {"Sns": 1000}

        for species in multi_species.species:
            if self.elliptical:
                active_population = environment_array[:, species.species_idx, 0]
                replacement_floor = active_population / species.deltat
                cap = replacement_caps.get(species.name)
                if cap is not None:
                    total_floor = np.sum(replacement_floor)
                    if total_floor > cap and total_floor > 0:
                        replacement_floor = replacement_floor * (cap / total_floor)
                solver_guess_array[:, species.species_idx, 0] = np.maximum(
                    solver_guess_array[:, species.species_idx, 0], replacement_floor
                )
            else:
                active_population = environment_array[species.start_slice:species.end_slice]
                replacement_floor = active_population / species.deltat
                cap = replacement_caps.get(species.name)
                if cap is not None:
                    total_floor = np.sum(replacement_floor)
                    if total_floor > cap and total_floor > 0:
                        replacement_floor = replacement_floor * (cap / total_floor)
                solver_guess_array[species.start_slice:species.end_slice] = np.maximum(
                    solver_guess_array[species.start_slice:species.end_slice], replacement_floor
                )

        return solver_guess_array

    def iam_solver(self, scenario_name, MOCAT_config, simulation_name, grid_search=False):
        """
            The main function that runs the IAM solver.
        """
        self.grid_search = grid_search
        # Define the species that are part of the constellation and fringe
        multi_species_names = ["S", "Su", "Sns"]
        # multi_species_names = ["S"]

        # This will create a list of OPUSSpecies objects. 
        multi_species = MultiSpecies(multi_species_names)

        #########################
        ### CONFIGURE MOCAT MODEL
        #########################
        # if self.MOCAT is None:
        self.MOCAT, multi_species = configure_mocat(MOCAT_config, multi_species=multi_species, grid_search=self.grid_search)
        self.elliptical = self.MOCAT.scenario_properties.elliptical
        print(self.MOCAT.scenario_properties.x0)

        multi_species.get_species_position_indexes(self.MOCAT)
        multi_species.get_mocat_species_parameters(self.MOCAT) # abstract species level information, like deltat, etc. 

        model_horizon = self.MOCAT.scenario_properties.simulation_duration
        tf = np.arange(1, model_horizon + 1)
        # create a list of the years (int) using scenario_properties.start_date and scenario_properties.simulation_duration
        years = [int(self.MOCAT.scenario_properties.start_date.year) + i for i in range(self.MOCAT.scenario_properties.simulation_duration)]
    
        #################################
        ### CONFIGURE ECONOMIC PARAMETERS
        #################################
        
        # For each simulation - we will need to modify the base economic parameters for the species. 
        for species in multi_species.species:
            species.econ_params.modify_params_for_simulation(scenario_name)
            species.econ_params.calculate_cost_fn_parameters(species.Pm, scenario_name)            

        # For now make all satellites circular if elliptical
        if self.elliptical:
            # for species idx in multi_species.species, place all of their satellites in the first eccentricity bin
            for species in multi_species.species:
                # Sum all satellites across all eccentricity bins for this species
                total_satellites = np.sum(self.MOCAT.scenario_properties.x0[:, species.species_idx, :], axis=1)
                # Move all satellites to the first eccentricity bin (index 0)
                self.MOCAT.scenario_properties.x0[:, species.species_idx, 0] = total_satellites
                # Set all other eccentricity bins to zero
                self.MOCAT.scenario_properties.x0[:, species.species_idx, 1:] = 0

        # Flatten for circular orbits
        if not self.elliptical:     
            self.MOCAT.scenario_properties.x0 = self.MOCAT.scenario_properties.x0.T.values.flatten()

        # Solver guess is 5% of the current fringe satellites. Update The launch file. This essentially helps the optimiser, as it is not a random guess to start with. 
        # Lam should be the same shape as x0 and is full of None values for objects that are not launched. 
        solver_guess = self.MOCAT.scenario_properties.x0.copy()
        lam = np.full_like(self.MOCAT.scenario_properties.x0, None, dtype=object)
        if self.elliptical:
            for species in multi_species.species:
                # lam will be n_sma_bins x n_ecc_bins x n_alt_shells
                initial_guess = 0.05 * self.MOCAT.scenario_properties.x0[:, species.species_idx, 0]
                # Ensure all values are non-negative (bounds requirement)
                initial_guess = np.maximum(initial_guess, 0.0)
                # if sum of initial guess is 0, multiply each element by 10
                if np.sum(initial_guess) == 0:
                    initial_guess[:] = 5

                lam[:, species.species_idx, 0] = initial_guess
                solver_guess[:, species.species_idx, 0] = initial_guess
        else:
            for species in multi_species.species:            
                initial_guess = 0.05 * np.array(self.MOCAT.scenario_properties.x0[species.start_slice:species.end_slice])       
                # Ensure all values are non-negative (bounds requirement)
                initial_guess = np.maximum(initial_guess, 0.0)
                if np.sum(initial_guess) == 0:
                    initial_guess[:] = 5
                solver_guess[species.start_slice:species.end_slice] = initial_guess
                lam[species.start_slice:species.end_slice] = solver_guess[species.start_slice:species.end_slice]

        # solver_guess = self._apply_replacement_floor(solver_guess, self.MOCAT.scenario_properties.x0, multi_species)
        if self.elliptical:
            for species in multi_species.species:
                lam[:, species.species_idx, 0] = solver_guess[:, species.species_idx, 0]
        else:
            for species in multi_species.species:
                lam[species.start_slice:species.end_slice] = solver_guess[species.start_slice:species.end_slice]

        ############################
        ### SOLVE FOR THE FIRST YEAR
        # elp: 
        ############################c
        open_access = MultiSpeciesOpenAccessSolver(self.MOCAT, solver_guess, self.MOCAT.scenario_properties.x0, "linear", lam, multi_species, years, 0)

        # This is now the first year estimate for the number of fringe satellites that should be launched.
        launch_rate, col_probability_all_species, umpy, excess_returns, last_non_compliance = open_access.solver()
        # launch rate is 92
        # launch rate is 6075

        lam = insert_launches_into_lam(lam, launch_rate, multi_species, self.elliptical)
             
        ####################
        ### SIMULATION LOOP
        # For each year, take the previous state of the environment, 
        # then use the sovler to calculate the optimal launch rate. 
        ###################

        current_environment = self.MOCAT.scenario_properties.x0 # Starts as initial population, and is in then updated. 

        species_data = {sp: np.zeros((self.MOCAT.scenario_properties.simulation_duration, self.MOCAT.scenario_properties.n_shells)) for sp in self.MOCAT.scenario_properties.species_names}

        # Store the ror, collision probability and the launch rate 
        simulation_results = {}

        for time_idx in tf:

            print("Starting year ", years[time_idx])
            
            # tspan = np.linspace(tf[time_idx], tf[time_idx + 1], time_step) # simulate for one year 
            tspan = np.linspace(0, 1, 2)
            
            # Propagate the model and take the final state of the environment
            if self.elliptical:
                state_next_sma, state_next_alt = self.MOCAT.propagate(tspan, current_environment, lam, elliptical=self.elliptical, use_euler=True, step_size=0.01)
            else:
                state_next_path, _ = self.MOCAT.propagate(tspan, current_environment, lam, elliptical=self.elliptical) # state_next_path: circ = 12077 elp = alt = 17763, self.x0: circ = 17914, elp = 17914
                if len(state_next_path) > 1:
                    state_next_alt = state_next_path[-1, :]
                else:
                    state_next_alt = state_next_path 

            # Apply PMD (Post Mission Disposal) evaluation to remove satellites
            print(f"Before PMD - Total environment: {np.sum(state_next_alt)}")
            if self.elliptical:
                state_next_sma, state_next_alt, multi_species = evaluate_pmd_elliptical(state_next_sma, state_next_alt, multi_species, years[time_idx], 
                self.MOCAT.scenario_properties.density_model, self.MOCAT.scenario_properties.HMid, self.MOCAT.scenario_properties.eccentricity_bins,
                self.MOCAT.scenario_properties.R0_rad_km)
            else:
                state_next_alt, multi_species = evaluate_pmd(state_next_alt, multi_species)
            print(f"After PMD - Total environment: {np.sum(state_next_alt)}")

            environment_for_solver = state_next_sma if self.elliptical else state_next_alt

            # Record propagated environment data 
            for i, sp in enumerate(self.MOCAT.scenario_properties.species_names):
                # 0 based index 
                if self.elliptical:
                    # For elliptical orbits, propagated_environment is a 2D array (n_shells, n_species)
                    species_data[sp][time_idx - 1] = state_next_alt[:, i]
                else:
                    # For circular orbits, propagated_environment is a 1D array
                    species_data[sp][time_idx - 1] = state_next_alt[i * self.MOCAT.scenario_properties.n_shells:(i + 1) * self.MOCAT.scenario_properties.n_shells]

            # Fringe Equilibrium Controller
            start_time = time.time()
            # solver guess will be lam
            solver_guess = None
            open_access = MultiSpeciesOpenAccessSolver(self.MOCAT, solver_guess, environment_for_solver, "linear", lam, multi_species, years, time_idx)

            # Calculate solver_guess
            solver_guess = lam.copy()
            for species in multi_species.species:
                # Calculate the probability of collision based on the new position
                if self.elliptical:
                    # For elliptical orbits, we need to use the 3D SMA matrix for collision probability
                    collision_probability = open_access.calculate_probability_of_collision(state_next_alt, species.name)
                    # Rate of Return - use the 3D SMA matrix
                    rate_of_return = open_access.fringe_rate_of_return(state_next_sma, collision_probability, species)
                else:
                    # For circular orbits, use the 2D matrix
                    collision_probability = open_access.calculate_probability_of_collision(state_next_alt, species.name)
                    # Rate of Return
                    rate_of_return = open_access.fringe_rate_of_return(state_next_alt, collision_probability, species)
                
                if self.elliptical:
                    # For elliptical orbits, solver_guess is 3D, so we need to handle it differently
                    # We'll update the first eccentricity bin (index 0) for this species
                    solver_guess[:, species.species_idx, 0] = solver_guess[:, species.species_idx, 0] - solver_guess[:, species.species_idx, 0] * (rate_of_return - collision_probability)
                else:
                    solver_guess[species.start_slice:species.end_slice] = solver_guess[species.start_slice:species.end_slice] - solver_guess[species.start_slice:species.end_slice] * (rate_of_return - collision_probability)

            # Check if there are any economic parameters that need to change (e.g demand growth of revenue)
            # multi_species.increase_demand()

            # solver_guess = self._apply_replacement_floor(solver_guess, environment_for_solver, multi_species)
            open_access = MultiSpeciesOpenAccessSolver(self.MOCAT, solver_guess, environment_for_solver, "linear", lam, multi_species, years, time_idx)

            # Solve for equilibrium launch rates
            launch_rate, col_probability_all_species, umpy, excess_returns, last_non_compliance = open_access.solver()

            # Update the initial conditions for the next period
            lam = insert_launches_into_lam(lam, launch_rate, multi_species, self.elliptical)

            elapsed_time = time.time() - start_time
            print(f'Time taken for period {time_idx}: {elapsed_time:.2f} seconds')

            # Update the current environment
            if self.elliptical:
                current_environment = state_next_sma
            else:
                current_environment = state_next_alt

            # Save the results that will be used for plotting later
            simulation_results[time_idx] = {
                "ror": rate_of_return,
                "collision_probability": collision_probability,
                "launch_rate" : launch_rate, 
                "collision_probability_all_species": col_probability_all_species,
                "umpy": umpy, 
                "excess_returns": excess_returns,
                "non_compliance": last_non_compliance
            }
        
        if self.grid_search:
            return species_data
        else:
            PostProcessing(self.MOCAT, scenario_name, simulation_name, species_data, simulation_results, multi_species.species[0].econ_params, grid_search=False)
            return species_data
        
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
                    # "bond_100k",
                    # "bondrevenuegrowth_100k",
                    # "revenuegrowth_0k",
                    # "bond_200k",
                    # # # "bond_300k",
                    # "bond_800k",
                    # "bond_1200k",
                    # "bond_1600k",
                    # # # "bond_100k_25yr",
                    # # # # "bond_200k_25yr",
                    # # "bond_300k_25yr",
                    # # # # "bond_500k_25yr",
                    # "bond_800k_25yr",
                    # "bond_1200k_25yr",
                    # "bond_1600k_25yr",
                    # "tax_1",
                    # # "tax_2"
                ]
    
    MOCAT_config = json.load(open("./OPUS/configuration/multi_single_species.json"))

    simulation_name = "bond-test-new-bonds"

    iam_solver = IAMSolver()

    def get_total_species_from_output(species_data):
        totals = {}
        for species, data_array in species_data.items():
            if isinstance(data_array, np.ndarray):
                totals[species] = np.sum(data_array[-1])
        return totals

    # # no parallel processing
    for scenario_name in scenario_files:
        # in the original code - they seem to look at both the equilibrium and the feedback. not sure why. I am going to implement feedback first. 
        output = iam_solver.iam_solver(scenario_name, MOCAT_config, simulation_name, grid_search=False)
        # Get the total species from the output
        total_species = get_total_species_from_output(output)
        print(f"Total species for scenario {scenario_name}: {total_species}")

    # # Parallel Processing
    # with ThreadPoolExecutor() as executor:
    #     # Map process_scenario function over scenario_files
    #     results = list(executor.map(process_scenario, scenario_files, [MOCAT_config]*len(scenario_files), [simulation_name]*len(scenario_files)))
 
    # # if you just want to plot the results - and not re- run the simulation. You just need to pass an instance of the MOCAT model that you created. 
    multi_species_names = ["S","Su", "Sns"]
    # # multi_species_names = ["Sns"]
    multi_species = MultiSpecies(multi_species_names)
    MOCAT, _ = configure_mocat(MOCAT_config, multi_species=multi_species, grid_search=False)
    PlotHandler(MOCAT, scenario_files, simulation_name, comparison=True)

    # PlotHandler(iam_solver.get_mocat(), scenario_files, simulation_name, comparison=False)
