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
from utils.EconCalculations import EconCalculations
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from datetime import timedelta
import time
import os
import pandas as pd

def ensure_bond_config_files(bond_amounts, lifetimes, config_dir="./OPUS/configuration/"):
    """
    Ensure all bond configuration CSV files exist with correct content.
    """
    scenario_names = []
    
    # Create configuration directory if it doesn't exist
    os.makedirs(config_dir, exist_ok=True)
    
    for bond_amount in bond_amounts:
        for lifetime in lifetimes:
            # Generate filename
            if bond_amount >= 1000000:
                bond_k = bond_amount // 1000
                filename = f"bond_{bond_k}k_{lifetime}yr.csv"
            elif bond_amount >= 1000:
                bond_k = bond_amount // 1000
                filename = f"bond_{bond_k}k_{lifetime}yr.csv"
            else:
                filename = f"bond_{bond_amount}k_{lifetime}yr.csv"
            
            filepath = os.path.join(config_dir, filename)
            scenario_name = filename.replace('.csv', '')
            
            # Expected content
            expected_content = f"""parameter_type,parameter_name,parameter_value
econ,bond,{bond_amount}
econ,disposal_time,{lifetime}
"""
            
            # Check if file exists and has correct content
            file_needs_update = False
            
            if not os.path.exists(filepath):
                print(f"Creating missing file: {filename}")
                file_needs_update = True
            else:
                # Check if content is correct
                try:
                    with open(filepath, 'r') as f:
                        current_content = f.read()
                    
                    if current_content.strip() != expected_content.strip():
                        print(f"Updating file with incorrect content: {filename}")
                        file_needs_update = True
                    else:
                        print(f"File exists and is correct: {filename}")
                        
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
                    file_needs_update = True
            
            # Update file if needed
            if file_needs_update:
                try:
                    with open(filepath, 'w') as f:
                        f.write(expected_content)
                    print(f"Successfully created/updated: {filename}")
                except Exception as e:
                    print(f"Error writing file {filename}: {e}")
                    continue
            
            scenario_names.append(scenario_name)
    
    return scenario_names

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
        self.config = None

    @staticmethod
    def get_species_position_indexes(MOCAT, constellation_sats):
        """
            The MOCAT model works on arrays that are the number of shells x number of species.
            Often throughout the model, we see the original list being spliced. 
            This function returns the start and end slice of the species in the array.
        """
        constellation_sats_idx = MOCAT.scenario_properties.species_names.index(constellation_sats)
        constellation_start_slice = (constellation_sats_idx * MOCAT.scenario_properties.n_shells)
        constellation_end_slice = constellation_start_slice + MOCAT.scenario_properties.n_shells

        return constellation_start_slice, constellation_end_slice

    def iam_solver(self, scenario_name, MOCAT_config, simulation_name, multi_species_names, grid_search=False):
        """
            The main function that runs the IAM solver.
        """
        self.grid_search = grid_search
        # Define the species that are part of the constellation and fringe
        # multi_species_names = ["SA", "SB", "SC", "SuA", "SuB", "SuC"]
        multi_species_names = ["S", "Su", "Sns"]

        # This will create a list of OPUSSpecies objects. 
        multi_species = MultiSpecies(multi_species_names)

        #########################
        ### CONFIGURE MOCAT MODEL
        #########################
        self.MOCAT, multi_species = configure_mocat(MOCAT_config, multi_species=multi_species, grid_search=self.grid_search)
        self.elliptical = self.MOCAT.scenario_properties.elliptical
        print(self.MOCAT.scenario_properties.x0)

        model_horizon = self.MOCAT.scenario_properties.simulation_duration
        tf = np.arange(1, model_horizon + 1)
        # create a list of the years (int)
        years = [int(self.MOCAT.scenario_properties.start_date.year) + i for i in range(self.MOCAT.scenario_properties.simulation_duration)]
        years.insert(0, years[0] - 1)

        multi_species.get_species_position_indexes(self.MOCAT)
        multi_species.get_mocat_species_parameters(self.MOCAT) # abstract species level information, like deltat, etc. 

        current_environment = self.MOCAT.scenario_properties.x0 # Starts as initial population, and is in then updated. 
        species_data = {sp: {year: np.zeros(self.MOCAT.scenario_properties.n_shells) for year in years} for sp in self.MOCAT.scenario_properties.species_names}

        # update time 0 as the initial population
        initial_year = years[0] # Get the first year (e.g., 2016)

        if self.elliptical:
            x0_alt = self.MOCAT.scenario_properties.sma_ecc_mat_to_altitude_mat(self.MOCAT.scenario_properties.x0)
        
        for i, sp in enumerate(self.MOCAT.scenario_properties.species_names):
            if self.elliptical:
                species_data[sp][initial_year] = x0_alt[:, i]
            else:
                species_data[sp][initial_year] = self.MOCAT.scenario_properties.x0[sp]

    
        #################################
        ### CONFIGURE ECONOMIC PARAMETERS
        #################################
        
        # ADDED: EconCalculations setup
        econ_params_gen = EconParameters(self.econ_params_json, mocat=self.MOCAT)
        # Set initial_removal_cost to 0 since we are ignoring ADR
        econ_calculator = EconCalculations(econ_params_gen, initial_removal_cost=0) 
        
        # For each simulation - we will need to modify the base economic parameters for the species. 
        for species in multi_species.species:
            # Set a flag on the econ object if its name is in our bonded list
            if hasattr(self, 'bonded_species_names') and species.name in self.bonded_species_names:
                species.econ_params.is_bonded_species = True

            # Now, call modify_params_for_simulation
            species.econ_params.modify_params_for_simulation(scenario_name)
            species.econ_params.calculate_cost_fn_parameters(species.Pm, scenario_name, species.name)        
            # species.econ_params.update_congestion_costs(multi_species, self.MOCAT.scenario_properties.x0)    

        # For now make all satellites circular if elliptical
        if self.elliptical:
            for species in multi_species.species:
                total_satellites = np.sum(self.MOCAT.scenario_properties.x0[:, species.species_idx, :], axis=1)
                self.MOCAT.scenario_properties.x0[:, species.species_idx, 0] = total_satellites
                self.MOCAT.scenario_properties.x0[:, species.species_idx, 1:] = 0

        # Flatten for circular orbits
        if not self.elliptical:     
            self.MOCAT.scenario_properties.x0 = self.MOCAT.scenario_properties.x0.T.values.flatten()

        # Solver guess
        solver_guess = self.MOCAT.scenario_properties.x0.copy()
        lam = np.full_like(self.MOCAT.scenario_properties.x0, None, dtype=object)
        
        if self.elliptical:
            for species in multi_species.species:
                initial_guess = 0.05 * self.MOCAT.scenario_properties.x0[:, species.species_idx, 0]
                initial_guess = np.maximum(initial_guess, 0.0)
                if np.sum(initial_guess) == 0:
                    initial_guess[:] = 5
                lam[:, species.species_idx, 0] = initial_guess
                solver_guess[:, species.species_idx, 0] = initial_guess
        else:
            for species in multi_species.species:
                inital_guess = 0.05 * np.array(self.MOCAT.scenario_properties.x0[species.start_slice:species.end_slice])  
                if sum(inital_guess) == 0:
                    inital_guess[:] = 5
                solver_guess[species.start_slice:species.end_slice] = inital_guess
                lam[species.start_slice:species.end_slice] = solver_guess[species.start_slice:species.end_slice]

        if self.elliptical:
            for species in multi_species.species:
                lam[:, species.species_idx, 0] = solver_guess[:, species.species_idx, 0]
        else:
            for species in multi_species.species:
                lam[species.start_slice:species.end_slice] = solver_guess[species.start_slice:species.end_slice]

        # ADDED: Finding fringe and constellation slices
        constellation_sats_idx = None
        constellation_start_slice = None
        constellation_end_slice = None
        fringe_sats_idx = None
        fringe_start_slice = None
        fringe_end_slice = None
        
        for sp_object in multi_species.species:
            if sp_object.name.startswith('S') and not sp_object.name.startswith('Su') and not sp_object.name.startswith('Sns'):
                constellation_sats_idx = sp_object.species_idx
                constellation_start_slice = sp_object.start_slice
                constellation_end_slice = sp_object.end_slice
            if sp_object.name.startswith('Su'):
                fringe_sats_idx = sp_object.species_idx
                fringe_start_slice = sp_object.start_slice
                fringe_end_slice = sp_object.end_slice

        if fringe_start_slice is None:
            raise ValueError("Could not find any 'Su' prefixed species to determine fringe slices.")


        ############################
        ### SOLVE FOR THE FIRST YEAR
        ############################
        # UPDATED: Added fringe_start_slice, fringe_end_slice
        open_access = MultiSpeciesOpenAccessSolver(self.MOCAT, solver_guess, self.MOCAT.scenario_properties.x0, "linear", lam, multi_species, years, 0, fringe_start_slice, fringe_end_slice)

        launch_rate = open_access.solver()

        # ADDED: First-year economics processing
        econ_calculator.process_period_economics(
            num_actually_removed=0,
            current_environment=self.MOCAT.scenario_properties.x0,
            multi_species=multi_species,
            new_tax_revenue=float(open_access._last_total_revenue)
        )

        lam = insert_launches_into_lam(lam, launch_rate, multi_species, self.elliptical)
             
        ####################
        ### SIMULATION LOOP
        ###################
        simulation_results = {}

        for time_idx in tf:

            try:
                print("Starting year ", years[time_idx-1])
            except Exception as e:
                print("Starting year ", time_idx)

            tspan = np.linspace(0, 1, 2)
            
            # Propagate
            if self.elliptical:
                state_next_sma, state_next_alt = self.MOCAT.propagate(tspan, current_environment, lam, elliptical=self.elliptical, use_euler=True, step_size=0.01)
            else:
                state_next_path, _ = self.MOCAT.propagate(tspan, current_environment, lam, elliptical=self.elliptical)
                if len(state_next_path) > 1:
                    state_next_alt = state_next_path[-1, :]
                else:
                    state_next_alt = state_next_path 

            # Apply PMD
            print(f"Before PMD - Total environment: {np.sum(state_next_alt)}")
            if self.elliptical:
                if self.MOCAT.scenario_properties.density_model != "static_exp_func":
                    try:
                        density_model_name = self.MOCAT.scenario_properties.density_model.__name__
                    except AttributeError:
                        raise ValueError(f"Density model {self.MOCAT.scenario_properties.density_model} does not have a name property")
                state_next_sma, state_next_alt, multi_species = evaluate_pmd_elliptical(state_next_sma, state_next_alt, multi_species, 
                    years[time_idx-1], density_model_name, self.MOCAT.scenario_properties.HMid, self.MOCAT.scenario_properties.eccentricity_bins, 
                    self.MOCAT.scenario_properties.R0_rad_km)
            else:
                state_next_alt, multi_species = evaluate_pmd(state_next_alt, multi_species)
            print(f"After PMD - Total environment: {np.sum(state_next_alt)}")

            environment_for_solver = state_next_sma if self.elliptical else state_next_alt
            
            # Set to 0 since we are skipping ADR
            num_removed_this_period = 0 

            # Record propagated environment data 
            for i, sp in enumerate(self.MOCAT.scenario_properties.species_names):
                if self.elliptical:
                    species_data[sp][years[time_idx]] = state_next_alt[:, i]
                else:
                    species_data[sp][years[time_idx]] = state_next_alt[i * self.MOCAT.scenario_properties.n_shells:(i + 1) * self.MOCAT.scenario_properties.n_shells]
            
            # Fringe Equilibrium Controller
            start_time = time.time()
            solver_guess = None
            
            # UPDATED: Added fringe_start_slice, fringe_end_slice
            open_access = MultiSpeciesOpenAccessSolver(self.MOCAT, solver_guess, environment_for_solver, "linear", lam, multi_species, years, time_idx, fringe_start_slice, fringe_end_slice)

            # Update the solver_guess
            solver_guess = lam.copy()
            for species in multi_species.species:
                collision_probability = open_access.calculate_probability_of_collision(state_next_alt, species.name)

                if species.maneuverable:
                    maneuvers = open_access.calculate_maneuvers(state_next_alt, species.name)
                    cost = maneuvers * 10000 # $10,000 per maneuver
                    if self.elliptical:
                        rate_of_return = open_access.fringe_rate_of_return(state_next_sma, collision_probability, species, cost)
                    else:
                        rate_of_return = open_access.fringe_rate_of_return(state_next_alt, collision_probability, species, cost)
                else:
                    if self.elliptical:
                        rate_of_return = open_access.fringe_rate_of_return(state_next_sma, collision_probability, species)
                    else:
                        rate_of_return = open_access.fringe_rate_of_return(state_next_alt, collision_probability, species)
                
                if self.elliptical:
                    solver_guess[:, species.species_idx, 0] = solver_guess[:, species.species_idx, 0] - solver_guess[:, species.species_idx, 0] * (rate_of_return - collision_probability)
                else:
                    solver_guess[species.start_slice:species.end_slice] = solver_guess[species.start_slice:species.end_slice] - solver_guess[species.start_slice:species.end_slice] * (rate_of_return - collision_probability)

            # UPDATED: Added fringe_start_slice, fringe_end_slice
            open_access = MultiSpeciesOpenAccessSolver(self.MOCAT, solver_guess, environment_for_solver, "linear", lam, multi_species, years, time_idx, fringe_start_slice, fringe_end_slice)

            # Solve for equilibrium launch rates
            launch_rate = open_access.solver()

            # Update the initial conditions for the next period
            lam = insert_launches_into_lam(lam, launch_rate, multi_species, self.elliptical)

            elapsed_time = time.time() - start_time
            print(f'Time taken for period {time_idx}: {elapsed_time:.2f} seconds')

            # Update the current environment
            if self.elliptical:
                current_environment = state_next_sma
            else:
                current_environment = state_next_alt

            # ADDED: Process Economics to calculate welfare
            new_total_tax_revenue = float(open_access._last_total_revenue)

            welfare, leftover_revenue = econ_calculator.process_period_economics(
                num_actually_removed=num_removed_this_period,
                current_environment=current_environment,
                multi_species=multi_species, # <-- CHANGED
                new_tax_revenue=new_total_tax_revenue
            )
            # Read revenues for storage
            shell_revenue = open_access._last_tax_revenue.tolist()
            total_tax_revenue_for_storage = float(open_access._last_total_revenue)


            launch_rate_by_species = {}
            for sp in multi_species.species:
                launch_rate_by_species[sp.name] = launch_rate[sp.start_slice:sp.end_slice].tolist()

            # UPDATED: Added new economic data to results
            simulation_results[time_idx] = {
                "ror": rate_of_return.tolist(), # .tolist() for JSON
                "collision_probability": collision_probability.tolist(), # .tolist() for JSON
                "launch_rate" : launch_rate_by_species,
                "collision_probability_all_species": open_access._last_collision_probability,
                "umpy": open_access.umpy, 
                "excess_returns": open_access._last_excess_returns,
                "non_compliance": open_access._last_non_compliance, 
                "compliance": open_access._last_compliance,
                "maneuvers": open_access._last_maneuvers,
                "cost of maneuvers": open_access._last_cost,
                "rate_of_return": open_access._last_rate_of_return,
                # --- ADDED ---
                "tax_revenue_total": total_tax_revenue_for_storage,
                "tax_revenue_by_shell": shell_revenue,
                "welfare": welfare,
                "bond_revenue": np.sum(open_access.bond_revenue),
                "leftover_revenue": leftover_revenue
            }
        
        if self.grid_search:
            return species_data
        else:
            # Create a dictionary of econ_params for all species
            all_econ_params = {
                species.name: species.econ_params 
                for species in multi_species.species 
                if hasattr(species, 'econ_params') and species.econ_params is not None
            }
            
            PostProcessing(self.MOCAT, scenario_name, simulation_name, species_data, simulation_results, all_econ_params, grid_search=False)
            return species_data
        
    def get_mocat(self):
        return self.MOCAT

# UPDATED: Added multi_species_names parameter
def run_scenario(scenario_name, MOCAT_config, simulation_name, multi_species_names):
    """
    Create a new IAMSolver instance for each scenario, run the simulation,
    and return the result from get_mocat().
    """
    solver = IAMSolver()
    # Pass multi_species_names
    solver.iam_solver(scenario_name, MOCAT_config, simulation_name, multi_species_names)
    return solver.get_mocat()

def process_scenario(scenario_name, MOCAT_config, simulation_name, multi_species_names):
    """
    Wrapper function for parallel processing that includes multi_species_names.
    """
    return run_scenario(scenario_name, MOCAT_config, simulation_name, multi_species_names)


if __name__ == "__main__":
    baseline = False
    bond_amounts = [0, 100000, 200000, 500000, 750000, 1000000, 2000000] #, 1500000, 2000000]
    lifetimes = [25]
    
    # Ensure all bond configuration files exist with correct content
    print("Ensuring bond configuration files exist...")
    bond_scenario_names = ensure_bond_config_files(bond_amounts, lifetimes)
    
    # Generate complete scenario names list
    scenario_files = []
    if baseline:
        scenario_files.append("Baseline")
    scenario_files.extend(bond_scenario_names)
    config = {
        "scenario_files": scenario_files,
        "baseline": baseline,
        "bond_amounts": bond_amounts,
        "lifetimes": lifetimes
    }
    
    MOCAT_config = json.load(open("./OPUS/configuration/multi_single_species.json"))

    simulation_name = "extensive"
    # check if Results/{simulation_name} exists
    if not os.path.exists(f"./Results/{simulation_name}"):
        os.makedirs(f"./Results/{simulation_name}")

    iam_solver = IAMSolver()

    # multi_species_names = ["SA", "SB", "SC", "SuA", "SuB", "SuC"]
    # iam_solver.bonded_species_names = ["SA", "SB", "SuA", "SuB"]
    multi_species_names = ["S", "Su", "Sns"]

    def get_total_species_from_output(species_data):
        totals = {}
        for species, year_data in species_data.items():
            if isinstance(year_data, dict):
                # Get the latest year's data
                latest_year = max(year_data.keys())
                latest_data = year_data[latest_year]
                
                if isinstance(latest_data, np.ndarray):
                    # Sum the array values
                    totals[species] = np.sum(latest_data)
                elif hasattr(latest_data, 'sum'):
                    # Handle pandas Series
                    totals[species] = latest_data.sum()
                else:
                    # Fallback for other data types
                    totals[species] = float(latest_data) if isinstance(latest_data, (int, float)) else 0
            elif isinstance(year_data, np.ndarray):
                # Handle direct array input (backward compatibility)
                totals[species] = np.sum(year_data[-1])
        
        return totals

    # # no parallel processing
    # for scenario_name in scenario_files:
    #     # in the original code - they seem to look at both the equilibrium and the feedback. not sure why. I am going to implement feedback first. 
    #     output = iam_solver.iam_solver(scenario_name, MOCAT_config, simulation_name, grid_search=False)
    #     # Get the total species from the output
    #     total_species = get_total_species_from_output(output)
    #     print(f"Total species for scenario {scenario_name}: {total_species}")

    # # Parallel Processing
    with ThreadPoolExecutor() as executor:
        # Map process_scenario function over scenario_files
        results = list(executor.map(process_scenario, scenario_files, [MOCAT_config]*len(scenario_files), [simulation_name]*len(scenario_files), [multi_species_names]*len(scenario_files)))
 
    # # if you just want to plot the results - and not re- run the simulation. You just need to pass an instance of the MOCAT model that you created. 
    # multi_species_names = ["S","Su", "Sns"]
    # # multi_species_names = ["Sns"]
    multi_species = MultiSpecies(multi_species_names)
    MOCAT, _ = configure_mocat(MOCAT_config, multi_species=multi_species, grid_search=False)
    PlotHandler(MOCAT, scenario_files, simulation_name, comparison=True)

    # PlotHandler(iam_solver.get_mocat(), scenario_files, simulation_name, comparison=False)