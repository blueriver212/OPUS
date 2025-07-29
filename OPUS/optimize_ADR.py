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
from itertools import repeat

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
            econ_params.removal_cost = float(current_params[4])
            econ_params.tax = float(current_params[5])
            econ_params.bond = float(current_params[6]) if current_params[6] is not None else None
            if econ_params.bond == 0:
                econ_params.bond = None
            econ_params.ouf = float(current_params[7]) 

        elif not scenario_name.startswith("Baseline") and os.path.exists(f"./OPUS/configuration/{scenario_name}.csv"):
            # Fallback to reading from CSV if not in the parameter grid
            econ_params.modify_params_for_simulation(scenario_name)
        elif scenario_name.startswith("Baseline"):
            # Handle the baseline case
            econ_params.bond = None
            econ_params.tax = 0
            econ_params.ouf = 0
            econ_params.removal_cost = 5000000



        econ_params.calculate_cost_fn_parameters()
        
        adr_params = ADRParameters(self.adr_params_json, mocat=self.MOCAT)
        counter = 0

        # sammie addition
        # note: need to fix this and remove the unnecessary redundancies
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

            adr_params.adr_times = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
            
            if current_params:
            # If parameters are found in the grid, apply them
                adr_params.target_species = [current_params[1]]
                adr_params.target_shell = [current_params[2]]
                if current_params[3] > 1:
                    adr_params.n_remove = [current_params[3]] 
                    adr_params.remove_method = ["n"]
                elif current_params[3] < 1:
                    adr_params.p_remove = [current_params[3]]
                    adr_params.remove_method = ["p"]


            # if adr_params.target_species is not None:
            #     if ("B" in adr_params.target_species) or ("N_0.00141372kg" in adr_params.target_species):
            #         adr_params.target_shell = [7]
            #     elif "N_223kg" in adr_params.target_species:
            #         adr_params.target_shell = [5]
            #     elif "N_0.567kg" in adr_params.target_species:
            #         adr_params.target_shell = [7]
            if (adr_params.target_species is None) or (adr_params.target_species == "none"):
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

        #J- Last year tax revenue and removal cost initialization
        tax_revenue_lastyr = 0.0
        removal_cost = 5000000
        leftover_tax_revenue = 0.0
        money_bucket_2 = 0.0
        money_bucket_1 = 0.0

        for time_idx in tf:

            print("Starting year ", time_idx)
            
            #J- Tax Revenue read in
            total_tax_revenue = float(open_access._last_total_revenue)
            shell_revenue = open_access.last_tax_revenue.tolist()

            # tspan = np.linspace(tf[time_idx], tf[time_idx + 1], time_step) # simulate for one year 
            tspan = np.linspace(0, 1, 2)
            
            # State of the environment during a simulation loop
            fringe_initial_guess = None

            # Propagate the model and take the final state of the environment
            propagated_environment = self.MOCAT.propagate(tspan, current_environment, lam)
            propagated_environment = propagated_environment[-1, :] 
            propagated_environment = evaluate_pmd(propagated_environment, econ_params.comp_rate, self.MOCAT.scenario_properties.species['active'][1].deltat,
                                                    fringe_start_slice, fringe_end_slice, derelict_start_slice, derelict_end_slice, econ_params)

            # sammie addition: adding in removals left from econ-adr branch
            adr_params.removals_left  = int((money_bucket_2 + tax_revenue_lastyr)// removal_cost)

            if (econ_params.tax == 0 and econ_params.bond == 0 and econ_params.ouf == 0) or (econ_params.bond == None and econ_params.tax == 0 and econ_params.ouf == 0):
                adr_params.removals_left = 20
            
            before = propagated_environment.copy() 
            # sammie addition: runs the ADR function if the current year is one of the specified removal years
            adr_params.time = time_idx
            if ((adr_params.adr_times is not None) and (time_idx in adr_params.adr_times) and (len(adr_params.adr_times) != 0)):
                propagated_environment, num_removed = implement_adr2(propagated_environment,self.MOCAT,adr_params)
                counter = counter + 1
                removals[str(time_idx)] = num_removed
                print("ADR Counter: " + str(counter))
                print("Did you ever hear the tragedy of Darth Plagueis the Wise?")
            
            # leftover_tax_revenue = tax_revenue_lastyr - (before - propagated_environment).sum()*removal_cost

            # if leftover_tax_revenue >= 0:
            #     leftover_tax_revenue = tax_revenue_lastyr - (before - propagated_environment).sum()*removal_cost
            #     money_bucket_1 = money_bucket_2 + leftover_tax_revenue
            # else: 
            #     leftover_tax_revenue = 0
            #     money_bucket_1 = money_bucket_2 + tax_revenue_lastyr - (before - propagated_environment).sum()*removal_cost

            # print("Last year's revenue (used this year for removals):",tax_revenue_lastyr,"in year", time_idx)
            # print("Leftover revenue:",tax_revenue_lastyr - (before - propagated_environment).sum()*removal_cost, "in year", time_idx)
            # print("Leftover revenue being adding to welfare:", leftover_tax_revenue, "in year", time_idx)
            # print("Leftover Money Bucket:", money_bucket_1, "in year", time_idx)

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
                econ_params, lam, fringe_start_slice, fringe_end_slice, derelict_start_slice, derelict_end_slice,adr_params)
            
            collision_probability = open_access.calculate_probability_of_collision(propagated_environment)
            ror = open_access.fringe_rate_of_return(propagated_environment, collision_probability)

            # Calculate solver_guess
            solver_guess = lam[fringe_start_slice:fringe_end_slice] - lam[fringe_start_slice:fringe_end_slice] * (ror - collision_probability) * launch_mask

            open_access = OpenAccessSolver(
                self.MOCAT, solver_guess, launch_mask, propagated_environment, "linear",
                econ_params, lam, fringe_start_slice, fringe_end_slice, derelict_start_slice, derelict_end_slice, adr_params)

            # Solve for equilibrium launch rates
            launch_rate, col_probability_all_species, umpy, excess_returns = open_access.solver()

            #J- Tax Revenue read in (this one may be redundant, but the results don't get printed without it?)
            total_tax_revenue = float(open_access.last_total_revenue)
            shell_revenue = open_access.last_tax_revenue.tolist()
            
            # Update the initial conditions for the next period
            lam[fringe_start_slice:fringe_end_slice] = launch_rate

            elapsed_time = time.time() - start_time
            print(f'Time taken for period {time_idx}: {elapsed_time:.2f} seconds')

            # Update the current environment
            current_environment = propagated_environment

            #J- Adding in Economic Welfare
            fringe_pop = current_environment[fringe_start_slice:fringe_end_slice]
            total_fringe_sat = np.sum(fringe_pop)
            welfare = 0.5 * econ_params.coef * total_fringe_sat ** 2 + leftover_tax_revenue

            #J- This year's tax revenue + leftover tax revenue from this year's removals, used for next year's removals
            money_bucket_2 = money_bucket_1
            tax_revenue_lastyr = float(open_access._last_total_revenue)

            # Save the results that will be used for plotting later
            simulation_results[time_idx] = {
                "ror": ror,
                "collision_probability": collision_probability,
                "launch_rate" : launch_rate, 
                "collision_probability_all_species": col_probability_all_species,
                "umpy": umpy, 
                "excess_returns": excess_returns,
                "ICs": x0, # sammie addition
                "excess_returns": excess_returns,
                "tax_revenue_total": total_tax_revenue,
                "tax_revenue_by_shell": shell_revenue,
                "welfare": welfare,
                "bond_revenue":open_access.bond_revenue,
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

    def fit(self, target_species, target_shell, amount_remove, removal_cost, tax_rate, bond, ouf):
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
        solver = IAMSolver()
        MOCAT_config = json.load(open("./OPUS/configuration/three_species.json"))
        solver.params = params

        # with ThreadPoolExecutor() as executor:
        #     # Map process_scenario function over scenario_files
        #     results = list(executor.map(process_scenario, scenario_files, [MOCAT_config]*len(scenario_files), [simulation_name]*len(scenario_files), repeat(params)))

        # # setting up dictionaries with the results from the solver
        # for i, items in enumerate(results):
        #     adr_dict.update(results[i][1])
        #     welfare_dict.update(results[i][2])

        # # finding maximum welfare value and minimum UMPY value
        # best_welfare = max(welfare_dict.values())            
        # best_umpy = min(adr_dict.values())

        # # updating the parameter grid with UMPY and welfare values in each scenario, then saving the indices of the
        # # minimum UMPY and maximum welfare within the parameter grid
        # for k, v in adr_dict.items():
        #     for i, rows in enumerate(params):
        #         if k in rows:
        #             params[i][7] = v
        #             if v == best_umpy and k == params[i][0]:
        #                 umpy_scen = params[i][0]
        #                 umpy_idx = i

        # for k, v in welfare_dict.items():
        #     for i, rows in enumerate(params):
        #         if k in rows:
        #             params[i][8] = v
        #             if v == best_welfare and k == params[i][0]:
        #                 welfare_scen = params[i][0]
        #                 welfare_idx = i

        # # finding the parameters for the best UMPY and welfare scenarios
        # umpy_species = params[umpy_idx][1]
        # umpy_shell = params[umpy_idx][2]
        # umpy_am = params[umpy_idx][3]
        # umpy_rc = params[umpy_idx][4]
        # umpy_tax = params[umpy_idx][5]
        # umpy_bond = params[umpy_idx][6]
        # umpy_ouf = params[umpy_idx][7]

        # welfare_species = params[welfare_idx][1]
        # welfare_shell = params[welfare_idx][2]
        # welfare_am = params[welfare_idx][3]
        # welfare_rc = params[welfare_idx][4]
        # welfare_tax = params[welfare_idx][5]
        # welfare_bond = params[welfare_idx][6]
        # welfare_ouf = params[welfare_idx][7]

        # # saving parameter grid
        # if not os.path.exists(os.path.dirname(save_path)):
        #     os.makedirs(os.path.dirname(save_path))
        # with open(save_path, 'w') as json_file:
        #     json.dump(params, json_file, indent=4)

        # # saving best UMPY and welfare scenarios and the parameters used
        # if not os.path.exists(os.path.dirname(f"./Results/{simulation_name}/comparisons/best_params.json")):
        #     os.makedirs(os.path.dirname(f"./Results/{simulation_name}/comparisons/best_params.json"))
        # with open(f"./Results/{simulation_name}/comparisons/best_params.json", 'w') as json_file:
        #     json.dump([{"Best UMPY Scenario":umpy_scen, "Index":umpy_idx, "Species":umpy_species, "Shell":umpy_shell, "Amount Removed":umpy_am, "Removal Cost":umpy_rc, "Tax":umpy_tax, "Bond":umpy_bond, "OUF":umpy_ouf, "UMPY":best_umpy, "Welfare":params[umpy_idx][9]}, 
        #               {"Best Welfare Scenario":welfare_scen, "Index":welfare_idx, "Species":welfare_species, "Shell":welfare_shell, "Amount Removed":welfare_am, "Removal Cost":welfare_rc, "Tax":welfare_tax, "Bond":welfare_bond, "OUF":welfare_ouf, "UMPY":params[welfare_idx][8], "Welfare":best_welfare},
        #               {"Baseline Scenario":"Baseline", "Index":0, "Species":"None", "Shell":"None", "Amount Removed":"None", "Tax":"None", "Bond":"None", "OUF":"None", "UMPY":params[0][8], "Welfare":params[0][9]}], json_file, indent = 4) 

        # print("Best UMPY Achieved: " + str(best_umpy) + " with target species " + str(umpy_species) + " and " + str(umpy_am)+" removed in " + str(umpy_scen) + " scenario. ")
        # print("Best UMPY Index: ", umpy_idx)
        # print("Welfare in Best UMPY Scenario: ", params[umpy_idx][8])
        
        # print("Best Welfare Achieved: " + str(best_welfare) + " with target species " + str(welfare_species) + " and " + str(welfare_am) + " removed in " + str(welfare_scen) + " scenario. ")
        # print("Best Welfare Index: ", welfare_idx)
        # print("UMPY in Best Welfare Scenario: ", params[welfare_idx][7])

        # # potentially saving the names of only the best two scenarios for simulations
        # if umpy_scen == welfare_scen:
        #     scenario_files = ["Baseline", umpy_scen]
        # elif umpy_scen != welfare_scen:
        #     scenario_files = ["Baseline", welfare_scen, umpy_scen]
        # scenario_files = [umpy_scen, welfare_scen]

        return self, solver.MOCAT, scenario_files, best_umpy

def run_scenario(scenario_name, MOCAT_config, simulation_name):
    """
    Create a new IAMSolver instance for each scenario, run the simulation,
    and return the result from get_mocat().
    """
    solver = IAMSolver()
    solver.iam_solver(scenario_name, MOCAT_config, simulation_name)
    return solver.get_mocat()

def process_scenario(scenario_name, MOCAT_config, simulation_name, params):
    iam_solver = IAMSolver()
    iam_solver.params = params
    iam_solver.iam_solver(scenario_name, MOCAT_config, simulation_name)
    return iam_solver.get_mocat(), iam_solver.adr_dict, iam_solver.welfare_dict

if __name__ == "__main__":
    ####################
    ### 2. SCENARIO DEFINITIONS
    # Change to set scenarios and parallelization

    # Define the scenario to run. Store them in an array. Should be valid names of parameter set CSV files. 
    ## See examples in scenarios/parsets and compare to files named --parameters.csv for how to create new ones.
    scenario_files=[
                    # "Baseline",
                    # "p_05",
                    # "p_10",
                    # "p_15",
                    # "p_20",
                    # "p_25",
                    # "p_35",
                    # "p_50",
                    # "p_65",
                    # "p_75",
                    # "p_85",
                    # "p_90",
                    # "p_95",
                    # "p_100"
                    # "n_5",
                    # "n_10",
                    # "n_25",
                    # "n_35",
                    # "n_50",
                    # "n_75",
                    # "n_100",
                    # "n_150",
                    # "n_200",
                    # "n_250"
                    # "adr_b",
                    # "bond_0k_25yr",
                    # "bond_100k",
                    # # "bond_200k",
                    #"bond_300k",
                    # "bond_500k",
                    # "bond_800k",
                    # "bond_1600k",
                    # "bond_100k_25yr",
                    # # "bond_200k_25yr",
                    # # "bond_300k_25yr",
                    # "bond_500k_25yr",
                    # "bond_800k_25yr",
                    # "tax_1",00000
                    # "tax_2"
                ]
    
    MOCAT_config = json.load(open("./OPUS/configuration/three_species.json"))

    simulation_name = "20_shell_opt_test_n_223kg_full"

    iam_solver = IAMSolver()

    # no parallel processing
    # for scenario_name in scenario_files:
    #     # in the original code - they seem to look at both the equilibrium and the feedback. not sure why. I am going to implement feedback first. 
    #     iam_solver.iam_solver(scenario_name, MOCAT_config, simulation_name)

    # Parallel Processing
    # PlotHandler(iam_solver.get_mocat(), scenario_files, simulation_name)
    # params = []
    #with ThreadPoolExecutor() as executor:
         # Map process_scenario function over scenario_files
         #results = list(executor.map(process_scenario, scenario_files, [MOCAT_config]*len(scenario_files), [simulation_name]*len(scenario_files), params))

       
    # sammie addition: set up different parameter lists
    ts = ["N_223kg"]
    # tp = np.linspace(0, 0.5, num=2)
    tn = [20]
    tax = [0] #[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
    bond = [None] #[0,100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000]*1
    ouf = [0]*1
    target_shell = range(1,16) # last number should be the number of shells + 1
    rc = np.linspace(5000000, 5000000, num=1) # could also switch to range(x,y) similar to target_shell

    # sammie addition: running the "fit" function for "optimization" based on lower UMPY values
    opt, MOCAT, scenario_files, best_umpy = IAMSolver.fit(iam_solver, target_species=ts, target_shell=target_shell, amount_remove=tn, removal_cost=rc, tax_rate=tax, bond=bond, ouf=ouf)

    # PlotHandler(MOCAT, scenario_files, simulation_name, comparison=True)

    # if you just want to plot the results - and not re- run the simulation. You just need to pass an instance of the MOCAT model that you created. 
    MOCAT,_, _ = configure_mocat(MOCAT_config, fringe_satellite="Su")
    PlotHandler(MOCAT, scenario_files, simulation_name, comparison=True)

    # normalize umpy and welfare over same value or something? take average??? 
    # look into similar method for current optimization of just umpy
    # create loop to run for each value in target_species, then go through all of the p_remove and save the stuff to a json grid before moving on to the next thing
    # compare the umpy scores of them and save the best ones 