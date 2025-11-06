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
import time
from datetime import timedelta
import os
import pandas as pd

from utils.ADRParameters import ADRParameters
from utils.ADR import optimize_ADR_removal, implement_adr
from utils.EconCalculations import EconCalculations
from optimize_ADR import OptimizeADR, process_optimizer_scenario

def ensure_bond_config_files(bond_amounts, lifetimes, config_dir="./OPUS/configuration/"):
    """
    Ensure all bond configuration CSV files exist with correct content.
    
    Args:
        bond_amounts (list): List of bond amounts in dollars
        lifetimes (list): List of disposal times in years
        config_dir (str): Directory containing configuration files
    
    Returns:
        list: List of scenario names that were created/verified
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
        self.adr_params_json = None
        self.config = None

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
        # multi_species_names = ["SA", "SB", "SC", "SuA", "SuB", "SuC"]
        multi_species_names = ["S", "Su", "Sns"]
        # multi_species_names = ["S"]

        # This will create a list of OPUSSpecies objects. 
        multi_species = MultiSpecies(multi_species_names)

        #########################
        ### CONFIGURE MOCAT MODEL
        #########################
        # if self.MOCAT is None:
        self.MOCAT, multi_species = configure_mocat(MOCAT_config, multi_species=multi_species, grid_search=self.grid_search)
        self.elliptical = self.MOCAT.scenario_properties.elliptical # elp, x0 = 12517
        print(self.MOCAT.scenario_properties.x0)

        model_horizon = self.MOCAT.scenario_properties.simulation_duration
        tf = np.arange(1, model_horizon + 1)
        # create a list of the years (int) using scenario_properties.start_date and scenario_properties.simulation_duration
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
        
        # sammie addition:
        adr_times = [5, 10, 15, 20]
        econ_params_gen = EconParameters(self.econ_params_json, mocat=self.MOCAT)
        econ_params_gen.econ_params_for_ADR(scenario_name)
        econ_calculator = EconCalculations(econ_params_gen, initial_removal_cost=5000000)

        # For each simulation - we will need to modify the base economic parameters for the species. 
        for species in multi_species.species:
            species.econ_params.modify_params_for_simulation(scenario_name)
            species.econ_params.calculate_cost_fn_parameters(species.Pm, scenario_name)            
            # species.econ_params.update_congestion_costs(multi_species, self.MOCAT.scenario_properties.x0) 

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
                # if sum of initial guess is 0, multiply each element by 10
                if np.sum(initial_guess) == 0:
                    initial_guess[:] = 5
                lam[:, species.species_idx, 0] = initial_guess
                solver_guess[:, species.species_idx, 0] = initial_guess
        else:
            for species in multi_species.species:
                # if species.name == constellation_sat:
                #     continue
                # else:
                inital_guess = 0.05 * np.array(self.MOCAT.scenario_properties.x0[species.start_slice:species.end_slice])  
                # if sum of initial guess is 0, muliply each element by 10
                if sum(inital_guess) == 0:
                    inital_guess[:] = 5
                solver_guess[species.start_slice:species.end_slice] = inital_guess
                lam[species.start_slice:species.end_slice] = solver_guess[species.start_slice:species.end_slice]

        # solver_guess = self._apply_replacement_floor(solver_guess, self.MOCAT.scenario_properties.x0, multi_species)
        if self.elliptical:
            for species in multi_species.species:
                lam[:, species.species_idx, 0] = solver_guess[:, species.species_idx, 0]
        else:
            for species in multi_species.species:
                lam[species.start_slice:species.end_slice] = solver_guess[species.start_slice:species.end_slice]

        adr_params = ADRParameters(self.adr_params_json, mocat=self.MOCAT)
        adr_params.adr_parameter_setup(scenario_name)
        # for species in multi_species.species:
        #     if species.adr_params is None:
        #         species.adr_params.adr_parameter_setup(scenario_name)


        #Finding fringe and constellation slices
        constellation_sats_idx = None
        constellation_start_slice = None
        constellation_end_slice = None
        fringe_sats_idx = None
        fringe_start_slice = None
        fringe_end_slice = None
        
        # Loop over the list of species objects
        for sp_object in multi_species.species:
            if sp_object.name == 'S':
                constellation_sats_idx = sp_object.species_idx
                constellation_start_slice = sp_object.start_slice
                constellation_end_slice = sp_object.end_slice

            if sp_object.name == 'Su':
                fringe_sats_idx = sp_object.species_idx
                fringe_start_slice = sp_object.start_slice
                fringe_end_slice = sp_object.end_slice

        # Check to make sure we found the slices we need
        if fringe_start_slice is None:
            raise ValueError("Could not find 'Su' species to determine fringe slices.")
            
        ############################
        ### SOLVE FOR THE FIRST YEAR 
        ############################
        open_access = MultiSpeciesOpenAccessSolver(self.MOCAT, solver_guess, self.MOCAT.scenario_properties.x0, "linear", lam, multi_species, years, 0, fringe_start_slice, fringe_end_slice)

        # This is now the first year estimate for the number of fringe satellites that should be launched.
        # Solver returns a tuple of 5 variables
        launch_rate = open_access.solver()

        # sammie / joey addition: This populates the `total_funds_for_removals` available for the start of the simulation loop (Year 1).


        econ_calculator.process_period_economics(
            num_actually_removed=0,
            current_environment=self.MOCAT.scenario_properties.x0,
            fringe_slices=(fringe_start_slice, fringe_end_slice),
            new_tax_revenue=float(open_access._last_total_revenue)
        )


        lam = insert_launches_into_lam(lam, launch_rate, multi_species, self.elliptical)
             
        ####################
        ### SIMULATION LOOP
        # For each year, take the previous state of the environment, 
        # then use the sovler to calculate the optimal launch rate. 
        ####################
        # Store the ror, collision probability and the launch rate 
        simulation_results = {}

        for time_idx in tf:

            try:
                print("Starting year ", years[time_idx-1])
            except Exception as e:
                print("Starting year ", time_idx)
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
                 # c heck if density_model has name property
                if self.MOCAT.scenario_properties.density_model != "static_exp_dens_func":
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

            # # ----- ADR Section ---- # #
            adr_params.removals_left = econ_calculator.get_removals_for_current_period()
            num_removed_this_period = 0; # initialize counter for removed objects
            adr_params.time = time_idx
            environment_before_adr = environment_for_solver.copy()

            if ((adr_params.adr_times is not None) and (time_idx in adr_params.adr_times) and (len(adr_params.adr_times) != 0)):
                # environment_for_solver, ~ = implement_adr(environment_for_solver,self.MOCAT,adr_params)
                environment_for_solver, removal_dict = optimize_ADR_removal(environment_for_solver,self.MOCAT,adr_params)
                num_removed_this_period = (environment_before_adr - environment_for_solver).sum

            # Record propagated environment data 
            for i, sp in enumerate(self.MOCAT.scenario_properties.species_names):
                # 0 based index 
                if self.elliptical:
                    # For elliptical orbits, propagated_environment is a 2D array (n_shells, n_species)
                    species_data[sp][years[time_idx - 1]] = state_next_alt[:, i]
                else:
                    # For circular orbits, propagated_environment is a 1D array
                    species_data[sp][years[time_idx - 1]] = state_next_alt[i * self.MOCAT.scenario_properties.n_shells:(i + 1) * self.MOCAT.scenario_properties.n_shells]

            # Fringe Equilibrium Controller
            start_time = time.time()
            # solver guess will be lam
            solver_guess = None
            open_access = MultiSpeciesOpenAccessSolver(self.MOCAT, solver_guess, environment_for_solver, "linear", lam, multi_species, years, time_idx, fringe_start_slice, fringe_end_slice)

            # Calculate solver_guess
            solver_guess = lam.copy()
            for species in multi_species.species:
                # Calculate the probability of collision based on the new position
                collision_probability = open_access.calculate_probability_of_collision(state_next_alt, species.name)

                if species.maneuverable:
                    maneuvers = open_access.calculate_maneuvers(state_next_alt, species.name)
                    # cost = species.econ_params.return_congestion_costs(state_next_alt, self.x0)
                    cost = maneuvers * 10000 # $10,000 per maneuver
                    # Rate of Return
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

            # store the rate of return for this species
            # Check if there are any economic parameters that need to change (e.g demand growth of revenue)
            # multi_species.increase_demand()

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

            # # ---- Process Economics ---- # #
            new_total_tax_revenue = float(open_access._last_total_revenue)

            welfare, leftover_revenue = econ_calculator.process_period_economics(
                num_actually_removed=num_removed_this_period,
                current_environment=current_environment,
                fringe_slices=(fringe_start_slice, fringe_end_slice),
                new_tax_revenue=new_total_tax_revenue
            )

            # Read revenues for storage
            shell_revenue = open_access._last_tax_revenue.tolist()
            total_tax_revenue_for_storage = float(open_access._last_total_revenue)

            # Save the results that will be used for plotting later
            simulation_results[time_idx] = {
                "ror": rate_of_return,
                "collision_probability": collision_probability,
                "launch_rate" : launch_rate, 
                "collision_probability_all_species": open_access._last_collision_probability,
                "umpy": open_access.umpy, 
                "excess_returns": open_access._last_excess_returns,
                "non_compliance": open_access._last_non_compliance, 
                "maneuvers": open_access._last_maneuvers,
                "cost": open_access._last_cost,
                "rate_of_return": open_access._last_rate_of_return,
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
    baseline = True
    bond_amounts = [100000]
    lifetimes = [5]
    
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

    simulation_name = "sns_test_2"
    if not os.path.exists(f"./Results/{simulation_name}"):
        os.makedirs(f"./Results/{simulation_name}")

    iam_solver = IAMSolver()

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

    # no parallel processing
    # for scenario_name in scenario_files:
    #     # in the original code - they seem to look at both the equilibrium and the feedback. not sure why. I am going to implement feedback first. 
    #     output = iam_solver.iam_solver(scenario_name, MOCAT_config, simulation_name, grid_search=False)
    #     # Get the total species from the output
    #     total_species = get_total_species_from_output(output)
    #     print(f"Total species for scenario {scenario_name}: {total_species}")

    # # Parallel Processing
    with ThreadPoolExecutor() as executor:
        # Map process_scenario function over scenario_files
        results = list(executor.map(process_scenario, scenario_files, [MOCAT_config]*len(scenario_files), [simulation_name]*len(scenario_files)))
 
    # # sammie addition: running the optimizer version of IAM Solver for shell-switching
    # optimization_solver = OptimizeADR()

    # ts = ["N_223kg"]
    # # tp = np.linspace(0, 0.5, num=2)
    # tn = [1000]
    # tax = [0] #[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
    # bond = [0, 1000000] #[0,100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000]*1
    # ouf = [0]*1
    # target_shell = [12] # last number should be the number of shells + 1
    # rc = np.linspace(5000000, 5000000, num=1) # could also switch to range(x,y) similar to target_shell

    # # sammie addition: running the "fit" function for "optimization" based on lower UMPY values
    # opt, MOCAT, scenario_files, best_umpy = OptimizeADR.fit(optimization_solver, target_species=ts, target_shell=target_shell, amount_remove=tn, removal_cost=rc, tax_rate=tax, bond=bond, ouf=ouf)


    # # if you just want to plot the results - and not re- run the simulation. You just need to pass an instance of the MOCAT model that you created. 
    multi_species_names = ["S","Su", "Sns"]
    # # multi_species_names = ["Sns"]
    # multi_species_names = ["SA", "SB", "SC", "SuA", "SuB", "SuC"]
    multi_species = MultiSpecies(multi_species_names)
    MOCAT, _ = configure_mocat(MOCAT_config, multi_species=multi_species, grid_search=False)
    PlotHandler(MOCAT, scenario_files, simulation_name, comparison=True)