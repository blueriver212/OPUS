import numpy as np
from scipy.optimize import least_squares
import sympy as sp
from concurrent.futures import ProcessPoolExecutor
from pyssem.model import Model
from scipy.optimize import least_squares
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm
from .PostMissionDisposal import evaluate_pmd

class EconCalculations:
    def __init__(self, MOCAT: Model, solver_guess, launch_mask, x0, revenue_model, 
                 econ_params, lam, fringe_start_slice, fringe_end_slice, derelict_start_slice, derelict_end_slice):
        test = "test"
        self.coef = econ_params.coef

        # calculations for later i think
        self.leftover_tax_revenue = None

    def revenue_calc():
        # whatever goes into calculating revenue
        test = "test"
        # self.leftover_tax_revenue = 

    def welfare_calc(self):
        
        welfare = 0.5 * self.coef * total_fringe_sat ** 2 + leftover_tax_revenue

