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
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
import time

from utils.ADRParameters import ADRParameters
from utils.ADR import optimize_ADR_removal, implement_adr
from utils.EconCalculations import EconCalculations
import os
from itertools import repeat


class OptimizeADR:
    def __init__(self, params = []):
        self.output = None
        self.MOCAT = None
        self.econ_params_json = None
        self.pmd_linked_species = None
        self.adr_params_json = None
        self.params = params 
        self.umpy_score = None
        self.welfare_dict = {}

    def solve_year_zero(self, scenario_name, MOCAT_config, simulation_name, grid_search=False):
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
        self.elliptical = self.MOCAT.scenario_properties.elliptical # elp, x0 = 12517

        multi_species.get_species_position_indexes(self.MOCAT)
        multi_species.get_mocat_species_parameters(self.MOCAT) # abstract species level information, like deltat, etc. 
    
        shells = np.arange(1, self.MOCAT.scenario_properties.n_shells +1)
        if MOCAT_config['OPUS']['disposal_time'] == 5:
            mids = self.MOCAT.scenario_properties.HMid
            mids = [abs(x - 400) for x in mids]
            natrually_compliant_idx = mids.index(min(mids))
            not_naturally_compliant_shells = shells[(natrually_compliant_idx+1):-1]
        elif MOCAT_config['OPUS']['disposal_time'] == 25:
            mids = self.MOCAT.scenario_properties.HMid
            mids = [abs(x - 520) for x in mids]
            natrually_compliant_idx = mids.index(min(mids))
            not_naturally_compliant_shells = shells[(natrually_compliant_idx+1):-1]

        #################################
        ### CONFIGURE ECONOMIC PARAMETERS
        #################################
        econ_params = EconParameters(self.econ_params_json, mocat=self.MOCAT)

        # J-Rewrote the below
        # Check if a parameter grid is being used for the current scenario
        current_params = None
        if self.params is not None and len(self.params) > 0:
            # Find the parameters for the current scenario_name
            for p in self.params:
                if p[0] == scenario_name:
                    current_params = p
                    break

        if current_params:
            # If parameters are found in the grid, apply them
            econ_params.removal_cost = float(current_params[4])
            econ_params.tax = float(current_params[5])
            econ_params.bond = float(current_params[6]) if current_params[6] is not None else None
            if econ_params.bond == 0:
                econ_params.bond = None
            econ_params.ouf = float(current_params[7])

            econ_params.calculate_cost_fn_parameters()
            self.econ_params = econ_params
        
        # sammie addition:
        if current_params:
            econ_calculator = EconCalculations(self.econ_params, initial_removal_cost=5000000)
        else:
            self.econ_params = EconParameters(self.econ_params_json, mocat=self.MOCAT)
            self.econ_params_gen.econ_params_for_ADR(scenario_name)
            econ_calculator = EconCalculations(self.econ_params_gen, initial_removal_cost=5000000)


        # For each simulation - we will need to modify the base economic parameters for the species. 
        for species in multi_species.species:
            species.econ_params.modify_params_for_simulation(scenario_name)
            species.econ_params.calculate_cost_fn_parameters(species.Pm, scenario_name)            

        self.adr_params = ADRParameters(self.adr_params_json, mocat=self.MOCAT)
        self.adr_params.adr_parameter_setup(scenario_name)
        self.adr_params.adr_times = np.arange(2, self.MOCAT.scenario_properties.simulation_duration+1)
            
        
        if current_params:
        # If parameters are found in the grid, apply them
            self.adr_params.target_species = [current_params[1]]
            self.adr_params.target_shell = [current_params[2]]
            self.adr_params.shell_order = [12, 14, 13, 15, 17, 11, 18, 16, 19, 20, 10, 9, 8, 5, 6, 7, 4, 3, 2, 1]
            if current_params[3] > 1:
                self.adr_params.n_remove = [current_params[3]] 
                self.adr_params.remove_method = ["n"]
            elif current_params[3] < 1:
                self.adr_params.p_remove = [current_params[3]]
                self.adr_params.remove_method = ["p"]


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
                inital_guess = 0.05 * n.array(self.MOCAT.scenario_properties.x0[species.start_slice:species.end_slice])  
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

        

        # for species in multi_species.species:
        #     if species.adr_params is None:
        #         species.adr_params.adr_parameter_setup(scenario_name)

        ############################
        ### SOLVE FOR THE FIRST YEAR
        # elp: 
        ############################c
        open_access = MultiSpeciesOpenAccessSolver(self.MOCAT, solver_guess, self.MOCAT.scenario_properties.x0, "linear", lam, multi_species, adr_params)

        # This is now the first year estimate for the number of fringe satellites that should be launched.
        launch_rate, col_probability_all_species, umpy, excess_returns, last_non_compliance = open_access.solver()
        # launch rate is 92
        # launch rate is 6075

        # sammie/ joey addition: This populates the `total_funds_for_removals` available for the start of the simulation loop (Year 1).
        econ_calculator.process_period_economics(
            num_actually_removed=0,
            current_environment=self.MOCAT.scenario_properties.x0,
            fringe_slices=(fringe_start_slice, fringe_end_slice),
            new_tax_revenue=float(open_access.last_total_revenue)
        )

        lam = insert_launches_into_lam(lam, launch_rate, multi_species, self.elliptical)
             
        ####################
        ### SIMULATION LOOP
        # For each year, take the previous state of the environment, 
        # then use the sovler to calculate the optimal launch rate. 
        ####################
        model_horizon = self.MOCAT.scenario_properties.simulation_duration
        tf = np.arange(1, model_horizon + 1) 

        current_environment = self.MOCAT.scenario_properties.x0 # Starts as initial population, and is in then updated. 

        species_data = {sp: np.zeros((self.MOCAT.scenario_properties.simulation_duration, self.MOCAT.scenario_properties.n_shells)) for sp in self.MOCAT.scenario_properties.species_names}

        
        return tf, current_environment, species_data, col_probability_all_species, umpy, excess_returns, last_non_compliance, econ_calculator, shells, lam

    def optimize_adr_loop(self, time_idx, species_data, econ_calculator, shells, current_environment, lam):
        current_trial_results = {}
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
            state_next_sma, state_next_alt, multi_species = evaluate_pmd_elliptical(state_next_sma, state_next_alt, multi_species)
        else:
            state_next_alt, multi_species = evaluate_pmd(state_next_alt, multi_species)
        print(f"After PMD - Total environment: {np.sum(state_next_alt)}")

        environment_for_solver = state_next_sma if self.elliptical else state_next_alt

        # # ----- ADR Section ---- # #
        removals_possible = econ_calculator.get_removals_for_current_period()
        self.adr_params.removals_left = removals_possible
        self.adr_params.time = time_idx
        if (self.econ_params.tax == 0 and self.econ_params.bond == 0 and self.econ_params.ouf == 0) or (econ_params.bond == None and econ_params.tax == 0 and econ_params.ouf == 0):
                self.adr_params.removals_left = 7 # This is a hard-coded override, kept as-is.

        before_adr = environment_for_solver.copy()
        self.environment_before_adr = before_adr.copy()

        for cs in shells:
            # reset environment and removals_left
            trial_environment_for_solver = before_adr.copy()
            self.adr_params.removals_left = removals_possible
            self.adr_params.target_shell = [cs]

            if ((self.adr_params.adr_times is not None) and (time_idx in self.adr_params.adr_times) and (len(adr_params.adr_times) != 0)):
                # environment_for_solver, ~ = implement_adr(environment_for_solver,self.MOCAT,adr_params)
                trial_environment_for_solver, removal_dict = optimize_ADR_removal(trial_environment_for_solver,self.MOCAT,adr_params)

            trial_num_removed = (before_adr - trial_environment_for_solver).sum
            trial_cost_of_removals = trial_num_removed * econ_calculator.removal_cost
            trial_funds_left = econ_calculator.total_funds_for_removals - trial_cost_of_removals
            trial_leftover_tax_revenue = max(0, trial_funds_left)

            # Record propagated environment data 
            trial_species_data = {}

            for i, sp in enumerate(self.MOCAT.scenario_properties.species_names):
                # 0 based index 
                if self.elliptical:
                    # For elliptical orbits, propagated_environment is a 2D array (n_shells, n_species)
                    trial_species_data[sp][time_idx - 1] = state_next_alt[:, i]
                else:
                    # For circular orbits, propagated_environment is a 1D array
                    trial_species_data[sp][time_idx - 1] = state_next_alt[i * self.MOCAT.scenario_properties.n_shells:(i + 1) * self.MOCAT.scenario_properties.n_shells]

            # Fringe Equilibrium Controller
            start_time = time.time()
            # solver guess will be lam
            solver_guess = None
            open_access = MultiSpeciesOpenAccessSolver(self.MOCAT, solver_guess, trial_environment_for_solver, "linear", lam, multi_species, adr_params=self.adr_params)

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

            solver_guess = self._apply_replacement_floor(solver_guess, trial_environment_for_solver, multi_species)
            open_access = MultiSpeciesOpenAccessSolver(self.MOCAT, solver_guess, trial_environment_for_solver, "linear", lam, multi_species, adr_params=self.adr_params)

            # Solve for equilibrium launch rates
            launch_rate, col_probability_all_species, umpy, excess_returns, last_non_compliance = open_access.solver()

            # Update the initial conditions for the next period
            # lam = insert_launches_into_lam(lam, launch_rate, multi_species, self.elliptical)
            # trial update for lam:
            trial_launch_rate = launch_rate.copy()

            elapsed_time = time.time() - start_time
            print(f'Time taken for period {time_idx}: {elapsed_time:.2f} seconds')

            # Update the current environment
            if self.elliptical:
                current_environment = state_next_sma
            else:
                current_environment = state_next_alt

            # # ---- Process Economics ---- # #
            trial_total_tax_revenue = float(open_access._last_total_revenue)
            trial_shell_revenue = open_access.last_tax_revenue.tolist()

            """NEED TO FIX THE WELFARE CALCULATIONS HERE"""
            # #J- Adding in Economic Welfare
            # fringe_pop = propagated_environment_trial[fringe_start_slice:fringe_end_slice]
            # total_fringe_sat = np.sum(fringe_pop)
            # # Use the trial's leftover revenue
            # welfare = 0.5 * econ_params.coef * total_fringe_sat ** 2 + leftover_tax_revenue_trial

            # Save the results that will be used for plotting later
            current_trial_results[cs] = {
                "environment": trial_environment_for_solver.copy(),
                "num_removed": trial_num_removed,
                "new_tax_revenue": trial_total_tax_revenue,
                "launch_rate": trial_launch_rate,
                "removal_dict": removal_dict,
                "species_data": trial_species_data,
                "welfare": welfare,
                "simulation_data": {
                    "ror": rate_of_return,
                    "collision_probability": collision_probability,
                    "launch_rate" : launch_rate, 
                    "collision_probability_all_species": col_probability_all_species,
                    "umpy": umpy, 
                    "excess_returns": excess_returns,
                    "non_compliance": last_non_compliance,
                    "tax_revenue_total": trial_total_tax_revenue,
                    "tax_revenue_by_shell": trial_shell_revenue,
                    "welfare": welfare,
                    "bond_revenue": open_access.bond_revenue,
                }
            }
            # Track the best shell as we go
            if welfare > best_welfare_so_far:
                best_welfare_so_far = welfare
                opt_shell = cs

        return current_trial_results, opt_shell

    def run_optimizer_loop(self, scenario_name, simulation_name, MOCAT_config):
        simulation_results = {}
        opt_path = {}
        tf, current_environment, species_data, col_probability_all_species, umpy, excess_returns, last_non_compliance, econ_calculator, shells, lam = OptimizeADR.solve_year_zero(self, scenario_name, MOCAT_config, simulation_name, grid_search=False)
        for time_idx in tf:
            optimization_trial_results, opt_shell = OptimizeADR.optimize_adr_loop(self, time_idx, species_data, econ_calculator, current_environment=current_environment, lam=lam, shells=shells)
            
            if opt_shell is not None:
                best_trial_results = optimization_trial_results[opt_shell]

                current_environment = best_trial_results['environment']
                num_actually_removed = best_trial_results['num_removed']
                new_tax_revenue = best_trial_results['new_tax_revenue']
                lam[fringe_start_slice:fringe_end_slice] = best_trial_results['launch_rate']

                # Now we call process_period_economics to finalize the year's state and prepare the funds for the next period.
                welfare, _ = econ_calculator.process_period_economics(
                    num_actually_removed,
                    current_environment,
                    (fringe_start_slice, fringe_end_slice),
                    new_tax_revenue
                        )
                # Save the results of the best trial
                simulation_results[time_idx] = best_trial_results['simulation_data']
                opt_path[str(time_idx)] = best_trial_results['removal_dict']
                # Update the species data with the best trial's data
                for i, sp in enumerate(self.MOCAT.scenario_properties.species_names):
                    species_data[sp][time_idx - 1] = best_trial_results['species_data'][sp]
            else:
                welfare, _ = econ_calculator.process_period_economics(
                    0,
                    self.environment_before_adr,
                    (fringe_start_slice, fringe_end_slice),
                    0,
                )
                pass 
            
        
        var = PostProcessing(self.MOCAT, scenario_name, simulation_name, species_data, simulation_results, self.econ_params)

        # sammie addition: storing the optimizable values and params
        self.umpy_score = var.umpy_score
        self.adr_dict = var.adr_dict
        self.welfare_dict[scenario_name] = welfare

        removal_save_path = f"./Results/{simulation_name}/{scenario_name}/removal_path.json"
        if not os.path.exists(os.path.dirname(removal_save_path)):
            os.makedirs(os.path.dirname(removal_save_path))
        with open(removal_save_path, 'w') as json_file:
            json.dump(opt_path, json_file, indent=4)

        opt_save_path = f"./Results/{simulation_name}/{scenario_name}/opt_comparison_values.json"
        if not os.path.exists(os.path.dirname(opt_save_path)):
            os.makedirs(os.path.dirname(opt_save_path))
        with open(opt_save_path, 'w') as json_file:
            json.dump(optimization_trial_results, json_file, indent=4)

    def get_mocat_from_optimizer(self):
        return self.MOCAT
    
    def grid_setup(self, simulation_name, target_species, target_shell, amount_remove, removal_cost, tax_rate, bond, ouf):
        test = "test"
        params = [None]*(len(target_species)*len(amount_remove)*len(tax_rate)*len(bond)*len(ouf)*len(target_shell)+1)
        scenario_files = ["Baseline"]
        # params = [None]*(len(target_species)*len(amount_remove)*len(tax_rate)*len(bond)*len(ouf))
        # scenario_files = []
        counter = 1
        save_path = f"./Results/{simulation_name}/comparisons/umpy_opt_grid.json"
        adr_dict = {}
        welfare_dict = {}
        best_umpy = None

        # running through each parameter to set up configurations
        params[0] = ["Baseline", "none", 1, 0, 5000000, 0, 0, 0, [], []]
        for i, sp in enumerate(target_species):
            for k, shell in enumerate(target_shell):
                for j, am in enumerate(amount_remove):
                    for ii, rc in enumerate(removal_cost):
                        for jj, tax in enumerate(tax_rate):
                            for kk, bn in enumerate(bond):
                                for fee in ouf:
                                    scenario_name = f"Scenario_{counter}"
                                    scenario_files.append(scenario_name)
                                    params[counter] = [scenario_name, sp, shell, am, rc, tax, bn, fee, [], []]
                                    counter = counter + 1

        # scenario_files.append("Baseline")
        # setting up solver and MOCAT configuration
        solver = OptimizeADR()
        MOCAT_config = json.load(open("./OPUS/configuration/three_species.json"))
        solver.params = params

        with ProcessPoolExecutor() as executor:
            # Map process_scenario function over scenario_files
            results = list(executor.map(process_optimizer_scenario, scenario_files, [MOCAT_config]*len(scenario_files), [simulation_name]*len(scenario_files), repeat(params)))

        # setting up dictionaries with the results from the solver
        for i, items in enumerate(results):
            adr_dict.update(results[i][1])
            welfare_dict.update(results[i][2])

        # finding maximum welfare value and minimum UMPY value
        best_welfare = max(welfare_dict.values())            
        best_umpy = min(adr_dict.values())

        # updating the parameter grid with UMPY and welfare values in each scenario, then saving the indices of the
        # minimum UMPY and maximum welfare within the parameter grid
        for k, v in adr_dict.items():
            for i, rows in enumerate(params):
                if k in rows:
                    params[i][7] = v
                    if v == best_umpy and k == params[i][0]:
                        umpy_scen = params[i][0]
                        umpy_idx = i

        for k, v in welfare_dict.items():
            for i, rows in enumerate(params):
                if k in rows:
                    params[i][8] = v
                    if v == best_welfare and k == params[i][0]:
                        welfare_scen = params[i][0]
                        welfare_idx = i

        # finding the parameters for the best UMPY and welfare scenarios
        umpy_species = params[umpy_idx][1]
        umpy_shell = params[umpy_idx][2]
        umpy_am = params[umpy_idx][3]
        umpy_rc = params[umpy_idx][4]
        umpy_tax = params[umpy_idx][5]
        umpy_bond = params[umpy_idx][6]
        umpy_ouf = params[umpy_idx][7]

        welfare_species = params[welfare_idx][1]
        welfare_shell = params[welfare_idx][2]
        welfare_am = params[welfare_idx][3]
        welfare_rc = params[welfare_idx][4]
        welfare_tax = params[welfare_idx][5]
        welfare_bond = params[welfare_idx][6]
        welfare_ouf = params[welfare_idx][7]

        # saving parameter grid
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        with open(save_path, 'w') as json_file:
            json.dump(params, json_file, indent=4)

        # saving best UMPY and welfare scenarios and the parameters used
        if not os.path.exists(os.path.dirname(f"./Results/{simulation_name}/comparisons/best_params.json")):
            os.makedirs(os.path.dirname(f"./Results/{simulation_name}/comparisons/best_params.json"))
        with open(f"./Results/{simulation_name}/comparisons/best_params.json", 'w') as json_file:
            json.dump([{"Best UMPY Scenario":umpy_scen, "Index":umpy_idx, "Species":umpy_species, "Shell":umpy_shell, "Amount Removed":umpy_am, "Removal Cost":umpy_rc, "Tax":umpy_tax, "Bond":umpy_bond, "OUF":umpy_ouf, "UMPY":best_umpy, "Welfare":params[umpy_idx][9]}, 
                      {"Best Welfare Scenario":welfare_scen, "Index":welfare_idx, "Species":welfare_species, "Shell":welfare_shell, "Amount Removed":welfare_am, "Removal Cost":welfare_rc, "Tax":welfare_tax, "Bond":welfare_bond, "OUF":welfare_ouf, "UMPY":params[welfare_idx][8], "Welfare":best_welfare},
                      {"Baseline Scenario":"Baseline", "Index":0, "Species":"None", "Shell":"None", "Amount Removed":"None", "Tax":"None", "Bond":"None", "OUF":"None", "UMPY":params[0][8], "Welfare":params[0][9]}], json_file, indent = 4) 

        print("Best UMPY Achieved: " + str(best_umpy) + " with target species " + str(umpy_species) + " and " + str(umpy_am)+" removed in " + str(umpy_scen) + " scenario. ")
        print("Best UMPY Index: ", umpy_idx)
        print("Welfare in Best UMPY Scenario: ", params[umpy_idx][8])
        
        print("Best Welfare Achieved: " + str(best_welfare) + " with target species " + str(welfare_species) + " and " + str(welfare_am) + " removed in " + str(welfare_scen) + " scenario. ")
        print("Best Welfare Index: ", welfare_idx)
        print("UMPY in Best Welfare Scenario: ", params[welfare_idx][7])

        # potentially saving the names of only the best two scenarios for simulations
        if umpy_scen == welfare_scen:
            scenario_files = ["Baseline", umpy_scen]
        elif umpy_scen != welfare_scen:
            scenario_files = ["Baseline", welfare_scen, umpy_scen]
        scenario_files = [umpy_scen, welfare_scen]

        return self, solver.MOCAT, scenario_files, best_umpy

def process_optimizer_scenario(scenario_name, MOCAT_config, simulation_name, params):
        iam_solver_optimize = OptimizeADR()
        iam_solver_optimize.params = params
        iam_solver_optimize.run_optimizer_loop(iam_solver_optimize, scenario_name, simulation_name, MOCAT_config)
        return iam_solver_optimize.get_mocat_from_optimizer(), iam_solver_optimize.adr_dict, iam_solver_optimize.welfare_dict
