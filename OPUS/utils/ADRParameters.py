
import numpy as np
from pyssem.model import Model
from pyssem.utils.drag.drag import densityexp
import pandas as pd
import matplotlib.pyplot as plt
import json

# this is mostly a copy of EconParameters.py with a few adjustments.

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
        self.exogenous = 0
        
    def adr_parameter_setup(self, configuration, baseline=False):
        file = open('./OPUS/configuration/adr_setup.json')
        adr = json.load(file)
        if (not configuration.startswith("Baseline")) and (configuration in adr):
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
            self.exogenous = params["exogenous"]

            # shell removal order/precedence (temp)
            self.shell_order = [12, 14, 13, 15, 17, 11, 18, 16, 19, 20, 10, 9, 8, 5, 6, 7, 4, 3, 2, 1]

        else:
            print("No ADR implemented. ")
            self.target_species = []
            self.adr_times = []
            self.target_shell = []
            
