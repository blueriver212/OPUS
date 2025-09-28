import numpy as np
from pyssem.model import Model
from pyssem.utils.drag.drag import densityexp
import pandas as pd
import matplotlib.pyplot as plt
import json

# this is mostly a copy of EconParameters.py with a few adjustments.
# currently it's not really being used since I couldn't figure it out, but the goal is to make 
# it so that you can have ADR separate from the initial model setup.

class ADRParameters:
    """
    trying to implement adr in a different way using a class and csv files

    It is initialized with default values for the parameters. 
    Then the calculate_cost_fn_parameters function is called to calculate the cost function parameters.
    """
    def __init__(self, adr_params_json, mocat: Model):
        # i cannot tell if this is doing anything tbh
        # Save MOCAT
        self.mocat = mocat

        # self.lift_price = 5000
        # params = adr_params_json.get("adr",adr_params_json)

        self.target_species = None
        self.adr_times = None
        self.n_max = None        
        self.remove_method = None
        self.time = None
        self.removals_left = None
        self.shell_order = None
        
    def find_adr_stuff(self, configuration, baseline=False):
        """
            This will modify the paramers for VAR and econ_parameters based on an input csv file. 
        """
        if not configuration.startswith("Baseline"):
            # read the csv file - must be in the configuration folder
            path = f"./OPUS/configuration/{configuration}.csv"

            # read the csv file
            parameters = pd.read_csv(path)

            for i, row in parameters.iterrows():
                parameter_type = row['parameter_type']
                parameter_name = row['parameter_name']
                parameter_value = row['parameter_value']

                # Modify the value based on parameter_type
                if parameter_type == 'adr':
                    if parameter_value == 1:
                        file = open('./OPUS/configuration/'+configuration+'_adr.json')

                # read the json file, should also be in the configuration folder
                        params = json.load(file)
                
                        self.target_species = params["target_species"]
                        self.adr_times = params["adr_times"]
                        self.properties = params["properties"]
                else:
                    print("No ADR implemented.")
        else:
            print("No ADR implemented. ")
        



        # if baseline:
        #     self.bond = None
        #     self.tax = 0
    def adr_parameter_setup(self, configuration, baseline=False):
        if not configuration.startswith("Baseline"):
            test = "test"
            file = open('./OPUS/configuration/adr_setup.json')
            adr = json.load(file)
            if configuration in adr:
                params = adr[configuration]
                self.adr_times = params["adr_times"]
                self.target_species = params["target_species"]
                
                self.remove_method = params["remove_method"]
                if "p" in self.remove_method:
                    self.p_remove = params["p_remove"]
                if "n" in self.remove_method:
                    self.n_remove = params["n_remove"]

                self.target_shell = params["target_shell"]
                self.n_max = params["n_max"]

                # shell removal order/precedence (temp)
                self.shell_order = [12, 14, 13, 15, 17, 11, 18, 16, 19, 20, 10, 9, 8, 5, 6, 7, 4, 3, 2, 1]

            elif configuration not in adr:
                print("No ADR implemented. ")
                self.target_species = []
                self.adr_times = []
                self.target_shell = []
        else:
            print("No ADR implemented. ")
            self.target_species = []
            self.adr_times = []
            self.target_shell = []
            
