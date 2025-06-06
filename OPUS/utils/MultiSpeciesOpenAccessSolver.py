import numpy as np
from scipy.optimize import least_squares
import sympy as sp
from concurrent.futures import ProcessPoolExecutor
from pyssem.model import Model
from scipy.optimize import least_squares
from joblib import Parallel, delayed
from tqdm import tqdm
from .PostMissionDisposal import evaluate_pmd
from .Helpers import insert_launches_into_lam

# sammie addition
from .ADR import implement_adr
from .ADR import implement_adr_cont

class MultiSpeciesOpenAccessSolver:
    def __init__(self, MOCAT: Model, solver_guess, launch_mask, x0, revenue_model, 
                 lam, multi_species, adr_params):
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
        self.launch_mask = launch_mask
        self.x0 = x0
        self.revenue_model = revenue_model
        self.lam = lam
        self.multi_species = multi_species

        self.tspan = np.linspace(0, 1, 2)
        self.time_idx = 0
        
        # sammie addition
        self.adr_params = adr_params

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
        # Add the fringe satellites to the lambda vector (this could also include constellation launches)
        # As the launches are only for the fringe, we need to add the launches to the correct slice of the lambda vector.
        
        self.lam = insert_launches_into_lam(self.lam, launches, self.multi_species)

        # Fringe_launches = self.fringe_launches # This will be the first guess by the model 
        state_next_path = self.MOCAT.propagate(self.tspan, self.x0, self.lam)
        state_next = state_next_path[-1, :]

        # Evaluate pmd
        state_next, multi_species = evaluate_pmd(state_next, self.multi_species)

        # sammie addition
        # Implement ADR
        # print("state_next_path: "+str(len(state_next_path)))
        state_next = implement_adr(state_next,self.MOCAT,self.adr_params)
        # state_next = implement_adr_cont(state_next, self.MOCAT, self.adr_params)

        # Gets the final output and update the current environment matrix
        self.current_environment = state_next

        # As excess returns is calculated on a per species basis, the launch array will need to be built.
        excess_returns = np.array([])
        for species in multi_species.species:
            # Calculate the probability of collision based on the new position
            collision_probability = self.calculate_probability_of_collision(state_next, species.name)

            # Rate of Return
            rate_of_return = self.fringe_rate_of_return(state_next, collision_probability, species)

            # Calculate the excess rate of return
            species_excess_returns= np.array((rate_of_return - collision_probability*(1 + species.econ_params.tax)) * 100)
            excess_returns = np.append(excess_returns, species_excess_returns)

        # Save the collision_probability for all species
        self._last_collision_probability = collision_probability
        self._last_excess_returns = excess_returns
        self._last_multi_species = multi_species

        non_compliance_dict = {
            species.name: species.sum_non_compliant for species in multi_species.species
        }

        self._last_non_compliance = non_compliance_dict

        return excess_returns

    def calculate_probability_of_collision(self, state_matrix, opus_species_name):
        """
            In the MOCAT Configuration, the indicated for active loss probability is already created. Now in the code, you just need to pass the state 
            matrix.

            Return: 
                - Active Loss per shell. This can be used to infer collision probability.  
        """
        evaluated_value = self.MOCAT.scenario_properties.fringe_active_loss[opus_species_name](*state_matrix)
        evaluated_value_flat = [float(value[0]) for value in evaluated_value]
        return np.array(evaluated_value_flat)
    
    def fringe_rate_of_return(self, state_matrix, collision_risk, opus_species):
        """
         Calcualtes the fringe rate of return. It can be only used by one species at once. 
         Currently it assumes a linear revenue model, although other models can be used in the future. 
        """
        fringe_total = state_matrix[opus_species.start_slice:opus_species.end_slice]

        revenue = opus_species.econ_params.intercept - opus_species.econ_params.coef * np.sum(fringe_total)
        # revenue = revenue * opus_species.launch_mask

        discount_rate = opus_species.econ_params.discount_rate

        depreciation_rate = 1 / opus_species.econ_params.sat_lifetime

        # Equilibrium expression for rate of return.
        rev_cost = revenue / opus_species.econ_params.cost

        if opus_species.econ_params.bond is None:
            rate_of_return = rev_cost - discount_rate - depreciation_rate  
        else:
            # bond_per_shell = self.econ_params.bond + (self.econ_params.bond * collision_risk)
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
        for species in self.multi_species.species:
            launch_rate_init = np.append(launch_rate_init, self.solver_guess[species.start_slice:species.end_slice])

        print(sum(launch_rate_init))

        # Define bounds for the solver
        lower_bound = np.zeros_like(launch_rate_init)  # Lower bound is zero

        # Define solver options
        solver_options = {
            'method': 'trf',  # Trust Region Reflective algorithm = trf
            'verbose': 0  # Show output if not parallelized
        }

        # Solve the system of equations
        result = least_squares(
            fun=lambda launches: self.excess_return_calculator(launches),
            x0=launch_rate_init,
            bounds=(lower_bound, np.inf),  # No upper bound
            **solver_options
        )

        # Extract the launch rate from the solver result, this will just be for the species
        launch_rate = result.x

        # if below 1, then change to 0 
        launch_rate[launch_rate < 1] = 0

        # Calculate the UMPY value
        umpy = self.MOCAT.opus_umpy_calculation(self.current_environment).flatten().tolist()

        return launch_rate, self._last_collision_probability, umpy, self._last_excess_returns, self._last_non_compliance