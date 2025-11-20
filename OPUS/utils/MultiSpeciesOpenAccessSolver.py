import numpy as np
from scipy.optimize import least_squares
import sympy as sp
from concurrent.futures import ProcessPoolExecutor
from pyssem.model import Model
from scipy.optimize import least_squares
from joblib import Parallel, delayed
from tqdm import tqdm
from .PostMissionDisposal import evaluate_pmd, evaluate_pmd_elliptical
from .Helpers import insert_launches_into_lam
from .EconCalculations import revenue_open_access_calculations

class MultiSpeciesOpenAccessSolver:
    # UPDATED __init__ signature
    def __init__(self, MOCAT: Model, solver_guess, x0, revenue_model, 
                 lam, multi_species, years, time_idx, fringe_start_slice, fringe_end_slice):
        """
        Initialize the MultiSpeciesOpenAccessSolver.
        """
        self.MOCAT = MOCAT
        self.solver_guess = solver_guess
        self.x0 = x0
        self.revenue_model = revenue_model
        self.lam = lam
        self.multi_species = multi_species
        self.elliptical = MOCAT.scenario_properties.elliptical
        self.tspan = np.linspace(0, 1, 2)
        self.time_idx = time_idx # <-- FIXED: Use passed time_idx
        self.years = years
        
        # ADDED: Initialize revenue variables
        self._last_total_revenue = 0.0
        self._last_tax_revenue = None
        self.bond_revenue = 0.0
        self.fringe_start_slice = fringe_start_slice
        self.fringe_end_slice = fringe_end_slice

        # This is the number of all objects in each shell. Starts as x0 (initial population)
        self.current_environment = x0 

        # This is temporary storage of each of the variables, so they can then be stored for visualisation later. 
        self._last_collision_probability = None
        self._last_maneuvers = None
        self._last_rate_of_return = None 
        self._last_non_compliance = None
        self._last_compliance = None

    def excess_return_calculator(self, launches):
        """
            Calculate the excess return for the given state matrix and launch rates.
        """

        self.lam = insert_launches_into_lam(self.lam, launches, self.multi_species, self.elliptical) 

        if self.elliptical:
            if self.MOCAT.scenario_properties.density_model == "static_exp_dens_func":
                state_next_sma, state_next_alt = self.MOCAT.propagate(self.tspan, self.x0, self.lam, self.elliptical, use_euler=True, step_size=0.05)
            else:
                state_next_sma, state_next_alt = self.MOCAT.propagate(self.tspan, self.x0, self.lam, self.elliptical, use_euler=True, step_size=0.05) 
        else:
            state_next_path, _ = self.MOCAT.propagate(self.tspan, self.x0, self.lam, elliptical=self.elliptical)
            if len(state_next_path) > 1:
                state_next_alt = state_next_path[-1, :]
            else:
                state_next_alt = state_next_path 

        # Evaluate pmd
        if self.elliptical:
            if self.MOCAT.scenario_properties.density_model != "static_exp_dens_func":
                try:
                    density_model_name = self.MOCAT.scenario_properties.density_model.__name__
                except AttributeError:
                    raise ValueError(f"Density model {self.MOCAT.scenario_properties.density_model} does not have a name property")
            state_next_sma, state_next_alt, multi_species = evaluate_pmd_elliptical(state_next_sma, state_next_alt, self.multi_species, 
                self.years[self.time_idx], density_model_name, self.MOCAT.scenario_properties.HMid, self.MOCAT.scenario_properties.eccentricity_bins, 
                self.MOCAT.scenario_properties.R0_rad_km)
        else:
            state_next_alt, multi_species = evaluate_pmd(state_next_alt, self.multi_species)

        # As excess returns is calculated on a per species basis
        excess_returns = {}
        collision_probability_dict = {}
        rate_of_return_dict = {}
        maneuvers_dict = {}
        cost_dict = {}

        for species in multi_species.species:
            collision_probability = self.calculate_probability_of_collision(state_next_alt, species.name)

            if species.maneuverable:
                maneuvers = self.calculate_maneuvers(state_next_alt, species.name)
                cost = maneuvers * 10000 * 2 # $10,000 per maneuver
                
                if self.elliptical:
                    rate_of_return = self.fringe_rate_of_return(state_next_sma, collision_probability, species, cost)
                else:
                    rate_of_return = self.fringe_rate_of_return(state_next_alt, collision_probability, species, cost)
            else:
                if self.elliptical:
                    rate_of_return = self.fringe_rate_of_return(state_next_sma, collision_probability, species)
                else:
                    rate_of_return = self.fringe_rate_of_return(state_next_alt, collision_probability, species)

            # Calculate the excess rate of return
            species_excess_returns=(rate_of_return - collision_probability*(1 + species.econ_params.tax)) * 100
            
            excess_returns[species.name] = species_excess_returns
            collision_probability_dict[species.name] = collision_probability
            rate_of_return_dict[species.name] = rate_of_return
            if species.maneuverable: # Only store if maneuverable
                maneuvers_dict[species.name] = maneuvers
                cost_dict[species.name] = cost

        # Save the collision_probability for all species
        self._last_collision_probability = collision_probability_dict
        self._last_excess_returns = excess_returns
        self._last_multi_species = multi_species
        self._last_cost = cost_dict
        self._last_rate_of_return = rate_of_return_dict
        self._last_maneuvers = maneuvers_dict
        if self.elliptical:
            self._last_current_environment_alt = state_next_alt

        non_compliance_dict = {
            species.name: species.sum_non_compliant for species in multi_species.species
        }
        compliance_dict = {
            species.name: species.sum_compliant for species in multi_species.species
        }

        self._last_non_compliance = non_compliance_dict
        self._last_compliance = compliance_dict

        # ADDED: Call to calculate revenue
        # This function will set self._last_total_revenue, self._last_tax_revenue,
        # and self.bond_revenue by modifying the 'self' (open_access_inputs) object.
        (
            self._last_tax_revenue,
            self._last_total_revenue,
            _, _, _, _,
        ) = revenue_open_access_calculations(self, state_next=state_next_alt)
        
        # convert excess_returns to a flattened numpy array
        excess_returns_flat = np.concatenate([excess_returns[species.name] for species in multi_species.species])
        return excess_returns_flat 

    def calculate_probability_of_collision(self, state_matrix, opus_species_name):
        """
            In the MOCAT Configuration, the indicated for active loss probability is already created. Now in the code, you just need to pass the state 
            matrix.
        """
        if self.elliptical:
            state_matrix = state_matrix.flatten()

        evaluated_value = self.MOCAT.scenario_properties.fringe_active_loss['collisions'][opus_species_name](*state_matrix)
        evaluated_value_flat = [float(value[0]) for value in evaluated_value]
        return np.array(evaluated_value_flat)

    def calculate_maneuvers(self, state_matrix, opus_species_name):
        """
            Calculates the maneuvers for the given state matrix and species name.
        """
        if self.elliptical:
            state_matrix = state_matrix.flatten()
        evaluated_value = self.MOCAT.scenario_properties.fringe_active_loss['maneuvers'][opus_species_name](*state_matrix)
        evaluated_value_flat = [float(value[0]) for value in evaluated_value]
        return np.array(evaluated_value_flat)
    
    def fringe_rate_of_return(self, state_matrix, collision_risk, opus_species, cost=None):
        """
         Calcualtes the fringe rate of return.
        """

        # Initialize total market population
        market_total_sum = 0.0
        
        if self.elliptical:
            # Get the population for the current species
            current_pop = state_matrix[:, opus_species.species_idx, 0]
            market_total_sum += np.sum(current_pop)

            # Check for competitors and add their totals
            if hasattr(opus_species.econ_params, 'competitors'):
                for competitor_name in opus_species.econ_params.competitors:
                    competitor_species = next((s for s in self.multi_species.species if s.name == competitor_name), None)
                    if competitor_species:
                        competitor_pop = state_matrix[:, competitor_species.species_idx, 0]
                        market_total_sum += np.sum(competitor_pop)
        else:
            # Get the population for the current species
            current_pop = state_matrix[opus_species.start_slice:opus_species.end_slice]
            market_total_sum += np.sum(current_pop)

            # Check for competitors and add their totals
            if hasattr(opus_species.econ_params, 'competitors'):
                for competitor_name in opus_species.econ_params.competitors:
                    competitor_species = next((s for s in self.multi_species.species if s.name == competitor_name), None)
                    if competitor_species:
                        competitor_pop = state_matrix[competitor_species.start_slice:competitor_species.end_slice]
                        market_total_sum += np.sum(competitor_pop)

        # The revenue calculation now correctly uses the total market sum
        revenue = opus_species.econ_params.intercept - opus_species.econ_params.coef * market_total_sum

        discount_rate = opus_species.econ_params.discount_rate
        depreciation_rate = 1 / opus_species.econ_params.sat_lifetime

        # Equilibrium expression for rate of return.
        base_cost = opus_species.econ_params.cost
        if cost is not None:
            total_cost = base_cost + cost
        else:
            total_cost = base_cost
        rev_cost = revenue / total_cost
      
        if opus_species.econ_params.bond is None:
            rate_of_return = rev_cost - discount_rate - depreciation_rate  
        else:
        #Updated the below the annualize the cost of the bond, in line with other calculations
            bond_value = opus_species.econ_params.bond
            comp_rate = opus_species.econ_params.comp_rate

            expecte_eol_loss = (1-comp_rate)*bond_value
            bond_cost_rate = (expecte_eol_loss)/total_cost

            rate_of_return = rev_cost - discount_rate - depreciation_rate - bond_cost_rate

        return rate_of_return
    
    def solver(self):
        """
        Solve the open-access launch rates.
        """
        launch_rate_init = np.array([])

        if self.elliptical:
            total_sats = {}
            for species in self.multi_species.species:
                sats_per_sma_bin = self.solver_guess[:, species.species_idx, 0]
                launch_rate_init = np.append(launch_rate_init, sats_per_sma_bin)
                total_sats[species.name] = np.sum(sats_per_sma_bin)

            print('Sats at start of elliptical solver: Total Sats', total_sats)
        else:
            for species in self.multi_species.species:
                launch_rate_init = np.append(launch_rate_init, self.solver_guess[species.start_slice:species.end_slice])
            print('Sats at start of circular solver: Total Sats', np.sum(launch_rate_init))
        
        if len(launch_rate_init) != self.MOCAT.scenario_properties.n_shells * len(self.multi_species.species):
            raise ValueError('Length of launch_rate_init is not the same as the number of shells * number of species in multi_species')
        
        lower_bound = np.zeros_like(launch_rate_init)

        solver_options = {
            'method': 'trf',
            'verbose': 0,
            'ftol': 5e-3,
            'xtol': 0.005,
            'gtol': 1e-3,
        }

        result = least_squares(
            fun=lambda launches: self.excess_return_calculator(launches),
            x0=launch_rate_init,
            bounds=(lower_bound, np.inf),
            **solver_options
        )

        launch_rate = result.x
        print(f"Launch rate: {launch_rate}")
        launch_rate[launch_rate < 1] = 0

        # Calculate the UMPY value
        if self.elliptical:
            state_for_umpy = self._last_current_environment_alt.flatten()
            self.umpy = self.MOCAT.opus_umpy_calculation(state_for_umpy).flatten().tolist()
        else:      
            self.umpy = self.MOCAT.opus_umpy_calculation(self.x0).flatten().tolist()

        return launch_rate