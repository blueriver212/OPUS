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

        # Save MOCAT
        self.mocat = mocat

        # self.lift_price = 5000
        params = adr_params_json.get("adr",adr_params_json)

        self.implement = params.get('implement',0)
        self.species = params.get('species',"B")
        self.shell = params.get('shell',[7,8])
        self.times = params.get('times',[5,10])
        self.p_remove = params.get('p_remove',0.2)
        
    def modify_adr_params_for_simulation(self, configuration, baseline=False):
        """
            This will modify the paramers for VAR and econ_parameters based on an input csv file. 
        """

        # read the csv file - must be in the configuration folder
        # file = open('./OPUS/configuration/{configuration}.json')
        file = open('./OPUS/configuration/'+configuration+'.json')

        # read the csv file
        params = json.load(file)
        
        self.implement = params["implement"]
        self.times = params["adr_times"]
        self.shell = params["target_shell"]
        self.species = params["target_species"]
        self.p_remove = params["p_remove"]


        # if baseline:
        #     self.bond = None
        #     self.tax = 0



