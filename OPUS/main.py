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

# sammie addition
from utils.ADRParameters import ADRParameters
from utils.ADR import implement_adr
from utils.ADR import implement_adr2
import os


class IAMSolver:

    def __init__(self, params = []):
        self.output = None
        self.MOCAT = None   
        self.econ_params_json = None
        self.pmd_linked_species = None
        # sammie addition
        self.adr_params_json = None
        self.params = params
        self.umpy_score = None
        self.welfare_dict = {}

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
            econ_params.tax = float(current_params[4])
            econ_params.bond = float(current_params[5]) if current_params[5] is not None else None
            econ_params.ouf = float(current_params[6]) 
        elif not scenario_name.startswith("Baseline") and os.path.exists(f"./OPUS/configuration/{scenario_name}.csv"):
            # Fallback to reading from CSV if not in the parameter grid
            econ_params.modify_params_for_simulation(scenario_name)
        elif scenario_name.startswith("Baseline"):
            # Handle the baseline case
            econ_params.bond = None
            econ_params.tax = 0
            econ_params.ouf = 0



        econ_params.calculate_cost_fn_parameters()
        
        adr_params = ADRParameters(self.adr_params_json, mocat=self.MOCAT)
        counter = 0

        # sammie addition
        adr_params.time = 0
        removals = {}
        if (self.params is None) or (len(self.params) == 0):
            adr_params.adr_parameter_setup(scenario_name)
        elif scenario_name.startswith("Baseline"):
            adr_params.target_species = []
            adr_params.p_remove = 0
            adr_params.remove_method = ["p"]
            adr_params.adr_times = []
        else:
            # setting up params for optimization
            if (self.params is not None) or len(self.params != 0):
                if scenario_name in self.params:
                    adr_params.target_species = [self.params[1]]
                    if self.params[3] < 1:
                        adr_params.p_remove = [self.params[3]]
                        adr_params.remove_method = ["p"]
                    elif self.params[3] > 1:
                        adr_params.n_remove = [self.params[3]]
                        adr_params.remove_method = ["n"]
                    adr_params.target_shell = [self.params[2]]
            else:
                adr_params.target_species = []
                adr_params.p_remove = 0
                adr_params.remove_method = ["p"]

            adr_params.adr_times = [3]
            
            if adr_params.target_species is not None:
                if ("B" in adr_params.target_species) or ("N_0.00141372kg" in adr_params.target_species):
                    adr_params.target_shell = [7]
                elif "N_223kg" in adr_params.target_species:
                    adr_params.target_shell = [5]
                elif "N_0.567kg" in adr_params.target_species:
                    adr_params.target_shell = [7]
            elif (adr_params.target_species is None) or (adr_params.target_species == "none"):
                adr_params.target_species = []
                adr_params.p_remove = 0
                adr_params.remove_method = ["p"]
        
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
                                    derelict_start_slice, derelict_end_slice,adr_params)

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

        # J- Initialize the economic calculator once
        econ_calculator = EconCalculations(econ_params, initial_removal_cost=5000000)

        for time_idx in tf:

            print("Starting year ", time_idx)
            
            # MOVED: These are now calculated and stored later
            # total_tax_revenue = float(open_access._last_total_revenue)
            # shell_revenue = open_access.last_tax_revenue.tolist()

            tspan = np.linspace(0, 1, 2)
            fringe_initial_guess = None

            # Propagate the model
            propagated_environment = self.MOCAT.propagate(tspan, current_environment, lam)
            propagated_environment = propagated_environment[-1, :] 
            propagated_environment = evaluate_pmd(propagated_environment, econ_params.comp_rate, self.MOCAT.scenario_properties.species['active'][1].deltat,
                                                fringe_start_slice, fringe_end_slice, derelict_start_slice, derelict_end_slice, econ_params)

            # --- ADR Section ---
            # J- Get the number of removals from our new econ calculations class class
            adr_params.removals_left = econ_calculator.get_removals_for_current_period()

            if (econ_params.tax == 0 and econ_params.bond == 0 and econ_params.ouf == 0) or (econ_params.bond is None and econ_params.tax == 0 and econ_params.ouf == 0):
                adr_params.removals_left = 20
            
            before = propagated_environment.copy() 
            num_removed_this_period = 0 # Initialize counter for removed objects
            adr_params.time = time_idx
            if ((adr_params.adr_times is not None) and (time_idx in adr_params.adr_times) and (len(adr_params.adr_times) != 0)):
                propagated_environment, num_removed_this_period = implement_adr2(propagated_environment, self.MOCAT, adr_params)
                counter = counter + 1
                removals[str(time_idx)] = num_removed_this_period
                print("ADR Counter: " + str(counter))

            # Update the constellation satellites for the next period
            for i in range(constellation_start_slice, constellation_end_slice):
                if lam[i] is not None:
                    lam[i] *= 0.05

            # Record propagated environment data
            for i, sp in enumerate(self.MOCAT.scenario_properties.species_names):
                species_data[sp][time_idx - 1] = propagated_environment[i * self.MOCAT.scenario_properties.n_shells:(i + 1) * self.MOCAT.scenario_properties.n_shells]

            # --- Fringe Equilibrium Controller Section ---
            start_time = time.time()
            print(f"Now starting period {time_idx}...")
            
            open_access = OpenAccessSolver(
                self.MOCAT, fringe_initial_guess, launch_mask, propagated_environment, "linear",
                econ_params, lam, fringe_start_slice, fringe_end_slice, derelict_start_slice, derelict_end_slice, adr_params)
            
            collision_probability = open_access.calculate_probability_of_collision(propagated_environment)
            ror = open_access.fringe_rate_of_return(propagated_environment, collision_probability)
            solver_guess = lam[fringe_start_slice:fringe_end_slice] - lam[fringe_start_slice:fringe_end_slice] * (ror - collision_probability) * launch_mask

            open_access = OpenAccessSolver(
                self.MOCAT, solver_guess, launch_mask, propagated_environment, "linear",
                econ_params, lam, fringe_start_slice, fringe_end_slice, derelict_start_slice, derelict_end_slice, adr_params)

            launch_rate, col_probability_all_species, umpy, excess_returns = open_access.solver()
            
            lam[fringe_start_slice:fringe_end_slice] = launch_rate
            elapsed_time = time.time() - start_time
            print(f'Time taken for period {time_idx}: {elapsed_time:.2f} seconds')
            
            current_environment = propagated_environment

            # --- Process Economics using the new class ---
            new_total_tax_revenue = float(open_access._last_total_revenue)
            
            welfare, leftover_revenue = econ_calculator.process_period_economics(
                num_actually_removed=num_removed_this_period,
                current_environment=current_environment,
                fringe_slices=(fringe_start_slice, fringe_end_slice),
                new_tax_revenue=new_total_tax_revenue
            )

            # --- Save Results ---
            # Read revenues for storage
            shell_revenue = open_access.last_tax_revenue.tolist()
            total_tax_revenue_for_storage = float(open_access._last_total_revenue)

            simulation_results[time_idx] = {
                "ror": ror,
                "collision_probability": collision_probability,
                "launch_rate": launch_rate,
                "collision_probability_all_species": col_probability_all_species,
                "umpy": umpy,
                "excess_returns": excess_returns,
                "ICs": x0,
                "tax_revenue_total": total_tax_revenue_for_storage, # Storing this period's generated tax
                "tax_revenue_by_shell": shell_revenue,
                "welfare": welfare, # Using the welfare calculated by our new class
                "bond_revenue": open_access.bond_revenue,
                "leftover_revenue": leftover_revenue # You can optionally store this too
            }

        var = PostProcessing(self.MOCAT, scenario_name, simulation_name, species_data, simulation_results, econ_params)

        # sammie addition: storing the optimizable values and params
        self.umpy_score = var.umpy_score
        self.adr_dict = var.adr_dict
        self.welfare_dict[scenario_name] = welfare

        save_path = f"./Results/{simulation_name}/{scenario_name}/objects_removed.json"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        with open(save_path, 'w') as json_file:
            json.dump([removals], json_file, indent=4)

    def get_mocat(self):
        return self.MOCAT

    def fit(self, target_species, amount_remove, tax_rate):
        # sammie addition:
        # function to create a "grid" of the specified parameters and plug those into the IAMSolver
        # Inputs:
        #   target_species -- a list of strings containing the names of the species being removed via ADR
        #   amount_remove -- a list of integers containing either the percentage or number of objects being removed through ADR
        #   tax_rate -- a list of integers containing the tax rate to be levied for ADR
        # Outputs:
        #   solver.MOCAT -- the configuration of MOCAT as used in the solver
        #   scenario_files -- the list of scenarios used in the optimization
        #   best_umpy -- the lowest UMPY value achieved in the scenarios

        params = [None]*(len(target_species)*len(amount_remove)*len(tax_rate))
        scenario_files = []
        counter = 0
        save_path = f"./Results/{simulation_name}/comparisons/umpy_opt_grid.json"
        adr_dict = {}
        welfare_dict = {}
        best_umpy = None

        # running through each parameter to set up configurations
        for i, sp in enumerate(target_species):
            for j, am in enumerate(amount_remove):
                for ii, tax in enumerate(tax_rate):
                    scenario_name = f"{sp}_{am}_{tax}"
                    scenario_files.append(scenario_name)
                    params[counter] = [scenario_name, sp, am, tax, [], []]
                    counter = counter + 1

        # setting up solver and MOCAT configuration
        solver = IAMSolver()
        MOCAT_config = json.load(open("./OPUS/configuration/three_species.json"))
        solver.params = params

        with ThreadPoolExecutor() as executor:
            # Map process_scenario function over scenario_files
            results = list(executor.map(process_scenario, scenario_files, [MOCAT_config]*len(scenario_files), [simulation_name]*len(scenario_files), params))

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
                    params[i][4] = v
                    if v == best_umpy and k == params[i][0]:
                        umpy_scen = params[i][0]
                        umpy_idx = i

        for k, v in welfare_dict.items():
            for i, rows in enumerate(params):
                if k in rows:
                    params[i][5] = v
                    if v == best_welfare and k == params[i][0]:
                        welfare_scen = params[i][0]
                        welfare_idx = i

        # finding the parameters for the best UMPY and welfare scenarios
        umpy_species = params[umpy_idx][1]
        umpy_am = params[umpy_idx][2]
        umpy_tax = params[umpy_idx][3]

        welfare_species = params[welfare_idx][1]
        welfare_am = params[welfare_idx][2]
        welfare_tax = params[welfare_idx][3]

        # saving parameter grid
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        with open(save_path, 'w') as json_file:
            json.dump(params, json_file, indent=4)

        # saving best UMPY and welfare scenarios and the parameters used
        if not os.path.exists(os.path.dirname(f"./Results/{simulation_name}/comparisons/best_params.json")):
            os.makedirs(os.path.dirname(f"./Results/{simulation_name}/comparisons/best_params.json"))
        with open(f"./Results/{simulation_name}/comparisons/best_params.json", 'w') as json_file:
            json.dump({"Best UMPY Scenario":umpy_scen, "Index":umpy_idx, "Species":umpy_species, "Amount Removed":umpy_am, "Tax Rate":umpy_tax, "UMPY":best_umpy, "Welfare":params[umpy_idx][5]}, 
                      {"Best Welfare Scenario":welfare_scen, "Index":welfare_idx, "Species":welfare_species, "Amount Removed":welfare_am, "Tax Rate":welfare_tax, "UMPY":params[welfare_idx][4], "Welfare":best_welfare}, json_file, indent = 4) 

        print("Best UMPY Achieved: " + str(best_umpy) + " with target species " + str(umpy_species) + " and " + str(umpy_am)+" removed and a tax rate of " + str(umpy_tax) + " in " + str(umpy_scen) + " scenario. ")
        print("Best UMPY Index: ", umpy_idx)
        print("Welfare in Best UMPY Scenario: ", params[umpy_idx][4])
        
        print("Best Welfare Achieved: " + str(best_welfare) + " with target species " + str(welfare_species) + " and " + str(welfare_am) + " removed and a tax rate of " + str(welfare_tax) + " in " + str(welfare_scen) + " scenario. ")
        print("Best Welfare Index: ", welfare_idx)
        print("UMPY in Best Welfare Scenario: ", params[welfare_idx][3])
        return self, solver.MOCAT, scenario_files, best_umpy

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
    # iam_solver.params = params
    iam_solver.iam_solver(scenario_name, MOCAT_config, simulation_name)
    return iam_solver.get_mocat(), iam_solver.adr_dict, iam_solver.welfare_dict

if __name__ == "__main__":
    ####################
    ### 2. SCENARIO DEFINITIONS
    # Change to set scenarios and parallelization

    # Define the scenario to run. Store them in an array. Should be valid names of parameter set CSV files. 
    ## See examples in scenarios/parsets and compare to files named --parameters.csv for how to create new ones.
    scenario_files=[
                    "Baseline",
                    # "Baseline_2",
                    # "Baseline_3",
                    # "Baseline_4",
                    # "Baseline_5",
                    # "Baseline_6",
                    # "Baseline_7",
                    # "Baseline_8",
                    # "Baseline_9",
                    # "Baseline_10",
                    # "adr_lnt",
                    # "25rule_Baseline",
                    "25rule_N223kg_cont",
                    "5rule_N223kg_cont",
                    "25rule_B_cont",
                    "5rule_B_cont",
                    "25rule_N0.5670kg_cont",
                    "5rule_N0.5670kg_cont",
                    "25rule_N0.00141372kg_cont",
                    "5rule_N0.00141372kg_cont",
                    "25rule_N223kg_one",
                    "5rule_N223kg_one",
                    "25rule_B_one",
                    "5rule_B_one",
                    "25rule_N0.5670kg_one",
                    "5rule_N0.5670kg_one",
                    "25rule_N0.00141372kg_one",
                    "5rule_N0.00141372kg_one",
                    # "bond_0k_25yr",
                    # "bond_100k",
                    # # "bond_200k",
                    # # "bond_300k",
                    # "bond_500k",
                    # "bond_800k",
                    # "bond_1600k",
                    # "bond_100k_25yr",
                    # # "bond_200k_25yr",
                    # # "bond_300k_25yr",
                    # "bond_500k_25yr",
                    # "bond_800k_25yr",
                    # "tax_1",
                    # "tax_2"
                ]
    
    MOCAT_config = json.load(open("./OPUS/configuration/three_species.json"))

    simulation_name = "25_year_vs_5_year_rules"

    iam_solver = IAMSolver()

    # no parallel processing
    # for scenario_name in scenario_files:
    #     # in the original code - they seem to look at both the equilibrium and the feedback. not sure why. I am going to implement feedback first. 
    #     iam_solver.iam_solver(scenario_name, MOCAT_config, simulation_name)

    # Parallel Processing
    # PlotHandler(iam_solver.get_mocat(), scenario_files, simulation_name)
    # with ThreadPoolExecutor() as executor:
    #     # Map process_scenario function over scenario_files
    #     results = list(executor.map(process_scenario, scenario_files, [MOCAT_config]*len(scenario_files), [simulation_name]*len(scenario_files)))


    # # sammie addition: set up different parameter lists
    # ts = ["N_223kg", "B"]
    # # tp = np.linspace(0, 0.5, num=2)
    # tn = np.linspace(0, 30, num=3)
    # tax = [0.17, 0.32]
    # # sammie addition: running the "fit" function for "optimization" based on lower UMPY values
    # # opt, MOCAT, scenario_files, best_umpy = IAMSolver.fit(iam_solver, target_species=ts, amount_remove=tn, tax_rate=tax)

    # PlotHandler(MOCAT, scenario_files, simulation_name, comparison = True)

    # if you just want to plot the results - and not re- run the simulation. You just need to pass an instance of the MOCAT model that you created. 
    MOCAT,_, _ = configure_mocat(MOCAT_config, fringe_satellite="Su")
    #PlotHandler(MOCAT, scenario_files, simulation_name, comparison=True)

    # normalize umpy and welfare over same value or something? take average??? 
    # look into similar method for current optimization of just umpy
    # create loop to run for each value in target_species, then go through all of the p_remove and save the stuff to a json grid before moving on to the next thing
    # compare the umpy scores of them and save the best ones 