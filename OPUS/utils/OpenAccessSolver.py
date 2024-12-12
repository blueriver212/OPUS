import numpy as np
from scipy.optimize import least_squares
import sympy as sp
from concurrent.futures import ProcessPoolExecutor
from pyssem.model import Model
import matplotlib.pyplot as plt

class OpenAccessSolver:
    def __init__(self, MOCAT: Model, state_matrix, launch_mask, x0, revenue_model, 
                 econ_params, lam, n_workers, fringe_index):
        """
        Initialize the OpenAccessSolver.

        Parameters:
            MOCAT: Instance of a MOCAT model
            state_matrix: Object containing variables for S, Su, N, D in all shells.
            revenue_model: Revenue model (e.g., 'linear').
            econ_params: Parameters for the revenue model.
            lam: Number of launches by the constellations.
            n_workers: Number of workers for parallel computing (default: 1).
        """
        self.MOCAT = MOCAT
        self.state_matrix = state_matrix
        self.launch_mask = launch_mask
        self.x0 = x0
        self.revenue_model = revenue_model
        self.econ_params = econ_params
        self.lam = lam
        self.fringe_index = fringe_index
        self.n_workers = n_workers
        self.tspan = np.linspace(0, 1, 2)

        # This is the number of all objects in each shell. Starts as x0 (initial population)
        self.current_environment = x0 

    def excess_return_calculator(self, launches):
        """
            Calculate the excess return for the given state matrix and launch rates.

            Launches: Open-access launch rates. This is just the fringe satellites. 1 x n_shells. 
        """
        
        # Calculate excess returns
        lam = self.lam

        # updates lam with up to date launches (from the solver)
        # Warning, MOCAT is expecting the launch file to be an array, with each item as a list. 
        for i in range(self.fringe_index * self.MOCAT.scenario_properties.n_shells, (self.fringe_index + 1) * self.MOCAT.scenario_properties.n_shells):
            lam[i] = [launches[i - self.fringe_index * self.MOCAT.scenario_properties.n_shells]]

        # fringe_launches = self.fringe_launches # This will be the first guess by the model 
        state_next_path = self.MOCAT.propagate(self.tspan, self.x0, lam)

        # gets the final output and update the current environment matrix
        state_next = state_next_path[-1, :]
        self.current_environment = state_next
        # Calculate the probability of collision based on the new positions
        collision_probability = self.calculate_probability_of_collision(state_next)

        rate_of_return = self.fringe_rate_of_return(state_next)

        # Calculate the excess rate of return
        excess_returns = 100 * (rate_of_return - collision_probability*(1 + self.econ_params.tax))

        return excess_returns

    def calculate_probability_of_collision(self, state_matrix):
        
        # THIS WILL BE IN THE MOCAT MODEL - NOT HERE
        # Currently the model is calcualting the probability of collision for each shell. I think this can be done for all shells at once. 
        # x0 is the current state of the system.

        # pass the x0 values to the equations
        values = [eq(*state_matrix) for eq in self.MOCAT.scenario_properties.coll_eqs_lambd]

        values = [values[i:i+self.MOCAT.scenario_properties.n_shells] for i in range(0, len(values), self.MOCAT.scenario_properties.n_shells)]

        su_only = values[1]

        collision_probability = 1 - np.exp(su_only)

        return collision_probability
    
    def fringe_rate_of_return(self, state_matrix):

        if self.revenue_model == "linear":
            # Linear revenue model: intercept - coef * total_number_of_fringe_satellites
            
            fringe_total = state_matrix[((self.fringe_index-1)*self.MOCAT.scenario_properties.n_shells):self.fringe_index*self.MOCAT.scenario_properties.n_shells]
            # print("Sum of sats: ", sum(fringe_total))
            revenue = self.econ_params.intercept - self.econ_params.coef * np.sum(fringe_total)
            revenue = revenue * self.launch_mask
            # print("Revenue: ",  sum(revenue))
            cost = self.econ_params.cost
            # print("Cost: ", cost)
            
            discount_rate = self.econ_params.discount_rate

            depreciation_rate = 1 / self.econ_params.sat_lifetime

            rate_of_return = revenue / cost - discount_rate - depreciation_rate  # Equilibrium expression for rate of return.

            # print("Rate of return",  rate_of_return)
        else:
            # Other revenue models can be implemented here
            rate_of_return = 0  # Placeholder value

        return rate_of_return
    
    def solve(self):
        return self.excess_return_calculator(self.state_matrix, self.lam)

        
    def solver(self, launch_rate_input, launch_mask):
        """
        Solve the open-access launch rates.

        Parameters: 
            launch_rate_input: Initial guess for open-access launch rates. 1 X n_shells, just the fringe satellites.
            launch_mask: Mask for the launch rates. Stops launches to certain altitudes if required. 

        Returns:
            numpy.ndarray: Open-access launch rates.
        """

        # Apply the launch mask to the initial guess
        launch_rate_init = launch_rate_input * launch_mask

        # Define bounds for the solver
        lower_bound = np.zeros_like(launch_rate_init)  # Lower bound is zero

        # Define solver options
        solver_options = {
            'method': 'trf',  #  Trust Region Reflective algorithm
            'verbose': 2 if self.n_workers == 1 else 0  # Show output if not parallelized
        }

        # Solve the system of equations
        result = least_squares(
            fun=lambda launches: self.excess_return_calculator(launches),
            x0=launch_rate_init,
            bounds=(lower_bound, np.inf),  # No upper bound
            **solver_options
        )

        # Extract the launch rate from the solver result
        launch_rate = result.x

        return launch_rate
    
    def find_initial_guess():
        """
            This function will estimate the first round of the fringe satellites that should be launched.

            For the optimization, it will require a first guess. This will be do 3 main things:
            1. Propagate the current population
            2. Calculate the rate of return
            3. Calculate the collision probability
            4. Provide an initial guess of fringe satellites

            Returns:
                numpy.ndarray: Initial guess of fringe satellites. 1 x n_shells.
        """

