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
class MultiSpeciesOpenAccessSolver:
    def __init__(self, MOCAT: Model, solver_guess, x0, revenue_model, 
                 lam, multi_species, years, time_idx):
        """
        Initialize the MultiSpeciesOpenAccessSolver.

        Parameters:
            MOCAT: Instance of a MOCAT model
            solver_guess: This is the initial guess of the fringe satellites. Array: 1 x n_shells.
            revenue_model: Revenue model (e.g., 'linear').
            econ_params: Parameters for the revenue model.
            lam: Number of launches by the constellations.
            multi_species: MultiSpecies object containing species information.
        """
        self.MOCAT = MOCAT
        self.solver_guess = solver_guess
        self.x0 = x0
        self.revenue_model = revenue_model
        self.lam = lam
        self.multi_species = multi_species
        self.elliptical = MOCAT.scenario_properties.elliptical
        self.tspan = np.linspace(0, 1, 2)
        self.time_idx = 0
        self.years = years

        # This is the number of all objects in each shell. Starts as x0 (initial population)
        self.current_environment = x0 

        # This is temporary storage of each of the variables, so they can then be stored for visualisation later. 
        self._last_collision_probability = None
        self._last_rate_of_return = None 
        self._last_non_compliance = None

    def excess_return_calculator(self, launches):
        """
            Calculate the excess return for the given state matrix and launch rates.

            Launches: Open-access launch rates. This is just the fringe satellites. 1 x n_shells. 
        """

        # print(f"Launches: {launches}")
        # Add the fringe satellites to the lambda vector (this could also include constellation launches)
        # As the launches are only for the fringe, we need to add the launches to the correct slice of the lambda vector.
        self.lam = insert_launches_into_lam(self.lam, launches, self.multi_species, self.elliptical) # circ = 92, elp = 92

        # Print total of self.lam, ignoring None values
        if self.elliptical:
            # For elliptical mode, lam is 3D: [n_sma_bins, n_species, n_ecc_bins]
            lam_total = np.sum(self.lam[self.lam != None])
        else:
            # For non-elliptical mode, lam is 1D
            lam_total = np.sum(self.lam[self.lam != None])
        print(f"Total lam (ignoring None): {lam_total}")

        # Fringe_launches = self.fringe_launches # This will be the first guess by the model 
        if self.elliptical:
            if self.MOCAT.scenario_properties.density_model == "static_exp_dens_func":
                state_next_sma, state_next_alt = self.MOCAT.propagate(self.tspan, self.x0, self.lam, self.elliptical, use_euler=True, step_size=0.05)
            else:
                state_next_sma, state_next_alt = self.MOCAT.propagate(self.tspan, self.x0, self.lam, self.elliptical, use_euler=True, step_size=0.05) #, density_year=self.years[self.time_idx])
        else:
            state_next_path, _ = self.MOCAT.propagate(self.tspan, self.x0, self.lam, elliptical=self.elliptical) # state_next_path: circ = 12077 elp = alt = 17763, self.x0: circ = 17914, elp = 17914
            if len(state_next_path) > 1:
                state_next_alt = state_next_path[-1, :]
            else:
                state_next_alt = state_next_path 
            # print(f"state_next_alt: {np.sum(state_next_alt)}")

        # Evaluate pmd
        if self.elliptical:
            # check if density_model has name property
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
        # 12077, elp = 18076

        # As excess returns is calculated on a per species basis, the launch array will need to be built.
        excess_returns = np.array([])
        collision_probability = np.array([])
        rate_of_return = np.array([])

        # For collision calculations and fringe rate of return, we are able to use the effective state matrix for elliptical orbits. 
        for species in multi_species.species:
            # Calculate the probability of collision based on the new position
            collision_probability = self.calculate_probability_of_collision(state_next_alt, species.name)
            # 2.1216498825008983e-05
            # 2.54775920739587e-05

            # Rate of Return
            rate_of_return = self.fringe_rate_of_return(self.current_environment, collision_probability, species)
            # array([0.074947  , 0.05510007, 0.0372975 , 0.02245892])
            # array([-0.60438682, -0.58274178, -0.56332631, -0.54714337])

            # Calculate the excess rate of return
            species_excess_returns= np.array((rate_of_return - collision_probability*(1 + species.econ_params.tax)) * 100)
            excess_returns = np.append(excess_returns, species_excess_returns)
            # array([7.49470045, 5.51000707, 3.72974955, 2.24377018])

        # Save the collision_probability for all species
        self._last_collision_probability = collision_probability
        self._last_excess_returns = excess_returns
        self._last_multi_species = multi_species

        print(excess_returns)

        non_compliance_dict = {
            species.name: species.sum_non_compliant for species in multi_species.species
        }

        self._last_non_compliance = non_compliance_dict

        return excess_returns 
        # circ: array([  7.42215529,   7.60501665,   6.6453673 ,   5.73031984,
        #      4.8622846 ,  28.42671294,  29.87327598,  26.4674035 ,
        #     22.68498412,  19.55166743, -43.66517506, -43.67581944,
        #    -43.72272862, -43.77695933, -43.95676902])
        # ellip: array([  7.500798  ,   7.68455259,   6.72716597,   5.84561885,
        #      5.01025917,  28.52251213,  29.9731395 ,  26.67021196,
        #     23.78536816,  21.27589047, -37.91651136, -37.61854428,
        #    -37.58261531, -37.50351418, -37.5294675 ])

    def calculate_probability_of_collision(self, state_matrix, opus_species_name):
        """
            In the MOCAT Configuration, the indicated for active loss probability is already created. Now in the code, you just need to pass the state 
            matrix.

            Return: 
                - Active Loss per shell. This can be used to infer collision probability.  
        """
        if self.elliptical:
            # For elliptical orbits, state_matrix is already a 2D altitude matrix
            # We need to convert it to the format expected by fringe_active_loss
            state_matrix = state_matrix.flatten()

        evaluated_value = self.MOCAT.scenario_properties.fringe_active_loss[opus_species_name](*state_matrix)
        evaluated_value_flat = [float(value[0]) for value in evaluated_value]
        return np.array(evaluated_value_flat)
    
    def fringe_rate_of_return(self, state_matrix, collision_risk, opus_species):
        """
         Calcualtes the fringe rate of return. It can be only used by one species at once. 
         Currently it assumes a linear revenue model, although other models can be used in the future. 
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
                    # Find the competitor species object
                    competitor_species = next((s for s in self.multi_species.species if s.name == competitor_name), None)
                    if competitor_species:
                        # Add the competitor's satellite count (elliptical)
                        competitor_pop = state_matrix[:, competitor_species.species_idx, 0]
                        market_total_sum += np.sum(competitor_pop)
        else:
            # Get the population for the current species
            current_pop = state_matrix[opus_species.start_slice:opus_species.end_slice]
            market_total_sum += np.sum(current_pop)

            # Check for competitors and add their totals
            if hasattr(opus_species.econ_params, 'competitors'):
                for competitor_name in opus_species.econ_params.competitors:
                    # Find the competitor species object
                    competitor_species = next((s for s in self.multi_species.species if s.name == competitor_name), None)
                    if competitor_species:
                        # Add the competitor's satellite count (circular)
                        competitor_pop = state_matrix[competitor_species.start_slice:competitor_species.end_slice]
                        market_total_sum += np.sum(competitor_pop)

        # The revenue calculation now correctly uses the total market sum
        revenue = opus_species.econ_params.intercept - opus_species.econ_params.coef * market_total_sum

        discount_rate = opus_species.econ_params.discount_rate
        # 0.05

        depreciation_rate = 1 / opus_species.econ_params.sat_lifetime
        # 0.2

        # Equilibrium expression for rate of return.
        rev_cost = revenue / opus_species.econ_params.cost
        # 1.189

        if opus_species.econ_params.bond is None:
            rate_of_return = rev_cost - discount_rate - depreciation_rate  
        else:
            bond_per_shell = np.ones_like(collision_risk) * opus_species.econ_params.bond
            bond = ((1-opus_species.econ_params.comp_rate) * (bond_per_shell / opus_species.econ_params.cost))
            rate_of_return = rev_cost - discount_rate - depreciation_rate - bond

        return rate_of_return
    
    def solver(self):
        """
        Solve the open-access launch rates.

        Parameters: 
            launch_rate_input: Initial guess for open-access launch rates. 1 X n_shells, just the fringe satellites.
            launch_mask: Mask for the launch rates. Stops launches to certain altitudes if required. 

        Returns:
            numpy.ndarray: Open-access launch rates.
        """
        # Make the launch rate only the length of the fringe satellites.
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
        
        # check that length of launch_rate_init is the same as the number of shells * number of species in multi_species
        if len(launch_rate_init) != self.MOCAT.scenario_properties.n_shells * len(self.multi_species.species):
            raise ValueError('Length of launch_rate_init is not the same as the number of shells * number of species in multi_species')
        
        # Define bounds for the solver
        lower_bound = np.zeros_like(launch_rate_init)  # Lower bound is zero
        # upper_bound = 100000 * np.ones_like(launch_rate_init)  # Upper bound is 100,000 satellites

        # Define solver options
        solver_options = {
            'method': 'trf',  # Trust Region Reflective algorithm = trf
            'verbose': 0,
            # 'ftol': 1e-8,   # Much tighter residual improvement threshold (was 5e-3)
            # 'xtol': 1e-8,   # Much tighter parameter convergence (was 0.05)
            # 'gtol': 1e-8,   # Much tighter gradient norm threshold (was 1e-3)
            # 'max_nfev': 1000  # Higher evaluation limit for stricter convergence
        }

        # Solve the system of equations
        # circ = ([ 5.,  5.,  5.,  5.,  5.,  0.,  0.,  5.,  6.,  4.,  0.,  2.,  5., 8., 13.])
        # array([ 5.,  5.,  5.,  5.,  5.,  0.,  0.,  5.,  6.,  4.,  0.,  2.,  5., 8., 13.])
        result = least_squares(
            fun=lambda launches: self.excess_return_calculator(launches),
            x0=launch_rate_init,
            bounds=(lower_bound, np.inf),  # No upper bound
            **solver_options
        )

        print(f" last excess returns: {self._last_excess_returns}")

        # Extract the launch rate from the solver result, this will just be for the species
        launch_rate = result.x

        print(f"Launch rate: {launch_rate}")

        # if below 1, then change to 0 
        launch_rate[launch_rate < 1] = 0

        # Calculate the UMPY value
        if self.elliptical:
            state_for_umpy = self.current_environment_alt.flatten()
            umpy = self.MOCAT.opus_umpy_calculation(state_for_umpy).flatten().tolist()
        else:      
            umpy = self.MOCAT.opus_umpy_calculation(self.current_environment).flatten().tolist()  # 120765

        return launch_rate, self._last_collision_probability, umpy, self._last_excess_returns, self._last_non_compliance