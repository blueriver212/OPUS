from utils.ConstellationParameters import ConstellationParameters
from utils.EconParameters import EconParameters
from utils.MocatParameters import configure_mocat   
from utils.OpenAccessSolver import OpenAccessSolver
from utils.PostProcessing import PostProcessing
from utils.PlotHandler import PlotHandler
from utils.PostMissionDisposal import evaluate_pmd
from utils.MultiSpecies import MultiSpecies
from utils.MultiSpeciesOpenAccessSolver import MultiSpeciesOpenAccessSolver
from utils.Helpers import insert_launches_into_lam
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
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
        self.MOCAT, multi_species = configure_mocat(MOCAT_config, multi_species=multi_species)
        print(self.MOCAT.scenario_properties.x0)

        multi_species.get_species_position_indexes(self.MOCAT)
        multi_species.get_mocat_species_parameters(self.MOCAT) # abstract species level information, like deltat, etc. 
        x0 = self.MOCAT.scenario_properties.x0.T.values.flatten()
    
        #################################
        ### CONFIGURE ECONOMIC PARAMETERS
        #################################
        
        # For each simulation - we will need to modify the base economic parameters for the species. 
        for species in multi_species.species:
            species.econ_params.modify_params_for_simulation(scenario_name)
            species.econ_params.calculate_cost_fn_parameters(species.Pm, scenario_name)            

        ############################
        ### CONSTELLATION PARAMETERS
        ############################

        # Get the slices for the constellation and fringe satellites
        # constellation_start_slice, constellation_end_slice = self.get_species_position_indexes(self.MOCAT, constellation_sat)
        # constellation_params = ConstellationParameters('./OPUS/configuration/constellation-parameters.csv')
        # lam = constellation_params.define_initial_launch_rate(self.MOCAT, constellation_start_slice, constellation_end_slice, x0)

        # Fringe population automomous controller. 
        launch_mask = np.ones((self.MOCAT.scenario_properties.n_shells,))

        # Solver guess is 5% of the current fringe satellites. Update The launch file. This essentially helps the optimiser, as it is not a random guess to start with. 
        solver_guess = x0
        lam = np.full_like(x0, None, dtype=object)
        for species in multi_species.species:
            # if species.name == constellation_sat:
            #     continue
            # else:
            inital_guess = 0.05 * np.array(x0[species.start_slice:species.end_slice]) * launch_mask  
            # if sum of initial guess is 0, muliply each element by 10
            if sum(inital_guess) == 0:
                inital_guess[:] = 5
            solver_guess[species.start_slice:species.end_slice] = inital_guess
            lam[species.start_slice:species.end_slice] = solver_guess[species.start_slice:species.end_slice]

        ############################
        ### SOLVE FOR THE FIRST YEAR
        ############################
        open_access = MultiSpeciesOpenAccessSolver(self.MOCAT, solver_guess, launch_mask, x0, "linear", lam, multi_species)

        # This is now the first year estimate for the number of fringe satellites that should be launched.
        launch_rate, col_probability_all_species, umpy, excess_returns, last_non_compliance = open_access.solver()

        lam = insert_launches_into_lam(lam, launch_rate, multi_species)
             
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
            propagated_environment = self.MOCAT.propagate(tspan, current_environment, lam, time_idx)
            propagated_environment = propagated_environment[-1, :] 
            propagated_environment, multi_species = evaluate_pmd(propagated_environment, multi_species)

            # Update the constellation satellites for the next period - should only be 5%.
            # for i in range(constellation_start_slice, constellation_end_slice):
            #     if lam[i] is not None:
            #         lam[i] = lam[i] * 0.05

            #lam = constellation_params.constellation_launch_rate_for_next_period(lam, sats_idx, x0, MOCAT)
        
            # Record propagated environment data
            for i, sp in enumerate(self.MOCAT.scenario_properties.species_names):
                # 0 based index 
                species_data[sp][time_idx - 1] = propagated_environment[i * self.MOCAT.scenario_properties.n_shells:(i + 1) * self.MOCAT.scenario_properties.n_shells]

            # Fringe Equilibrium Controller
            start_time = time.time()
            open_access = MultiSpeciesOpenAccessSolver(self.MOCAT, fringe_initial_guess, launch_mask, propagated_environment, "linear", lam, multi_species)

            # Calculate solver_guess
            solver_guess = lam
            for species in multi_species.species:
                # Calculate the probability of collision based on the new position
                collision_probability = open_access.calculate_probability_of_collision(propagated_environment, species.name)
                
                # Rate of Return
                rate_of_return = open_access.fringe_rate_of_return(propagated_environment, collision_probability, species)
                solver_guess[species.start_slice:species.end_slice] - solver_guess[species.start_slice:species.end_slice] * (rate_of_return - collision_probability) * launch_mask

            # Check if there are any economic parameters that need to change (e.g demand growth of revenue)
            multi_species.increase_demand()

            open_access = MultiSpeciesOpenAccessSolver(self.MOCAT, solver_guess, launch_mask, propagated_environment, "linear", lam, multi_species)

            # Solve for equilibrium launch rates
            launch_rate, col_probability_all_species, umpy, excess_returns, last_non_compliance = open_access.solver()

            # Update the initial conditions for the next period
            lam = insert_launches_into_lam(lam, launch_rate, multi_species)

            elapsed_time = time.time() - start_time
            print(f'Time taken for period {time_idx}: {elapsed_time:.2f} seconds')

            # Update the current environment
            current_environment = propagated_environment

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
                    # # "bond_200k",
                    # # # "bond_300k",
                    # # # # "bond_500k",
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

    simulation_name = "Single-Species-S"

    iam_solver = IAMSolver()

    def get_total_species_from_output(species_data):
        totals = {}
        for species, data_array in species_data.items():
            if isinstance(data_array, np.ndarray):
                totals[species] = np.sum(data_array[-1])
        return totals

    # # # no parallel processing
    for scenario_name in scenario_files:
        # in the original code - they seem to look at both the equilibrium and the feedback. not sure why. I am going to implement feedback first. 
        output = iam_solver.iam_solver(scenario_name, MOCAT_config, simulation_name, grid_search=True)
        # Get the total species from the output
        total_species = get_total_species_from_output(output)
        print(f"Total species for scenario {scenario_name}: {total_species}")

    # # Parallel Processing
    # with ThreadPoolExecutor() as executor:
    #     # Map process_scenario function over scenario_files
    #     results = list(executor.map(process_scenario, scenario_files, [MOCAT_config]*len(scenario_files), [simulation_name]*len(scenario_files)))
 
    # if you just want to plot the results - and not re- run the simulation. You just need to pass an instance of the MOCAT model that you created. 
    # multi_species_names = ["S","Su", "Sns"]
    # multi_species_names = ["Sns"]
    # multi_species = MultiSpecies(multi_species_names)
    # MOCAT, _ = configure_mocat(MOCAT_config, multi_species=multi_species)
    PlotHandler(MOCAT, scenario_files, simulation_name, comparison=True)


    # load in the multi_sing_species.json file
    single_json = json.load(open("./OPUS/configuration/multi_single_species.json"))
    names = ['name1', 'name2', 'name3']
    adr_values = [20, 40, 50]
    # then change the input json
    # then change the simulation name
    single_json['simulation_name'] = ['name1']

    for i in names:
        single_json['opus']['adr'] = adr_values[i]
        single_json['simulation_name'] = names[i]

        # run sim
        iam_solver.iam_solver(names[i], single_json, i, grid_search=False) 
    

    # create a grid space for adr values
    # some thing to solve for the best adr value
    # run through the adr values, take the output from iam_solver.iam_solver, store it, then use it for the next iteration 
