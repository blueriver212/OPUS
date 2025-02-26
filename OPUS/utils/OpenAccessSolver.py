import numpy as np
from scipy.optimize import least_squares
import sympy as sp
from concurrent.futures import ProcessPoolExecutor
from pyssem.model import Model
from scipy.optimize import least_squares
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm

class OpenAccessSolver:
    def __init__(self, MOCAT: Model, solver_guess, launch_mask, x0, revenue_model, 
                 econ_params, lam, fringe_start_slice, fringe_end_slice):
        """
        Initialize the OpenAccessSolver.

        Parameters:
            MOCAT: Instance of a MOCAT model
            solver_guess: This is the initial guess of the fringe satellites. Array: 1 x n_shells.
            revenue_model: Revenue model (e.g., 'linear').
            econ_params: Parameters for the revenue model.
            lam: Number of launches by the constellations.
            n_workers: Number of workers for parallel computing (default: 1).
        """
        self.MOCAT = MOCAT
        self.solver_guess = solver_guess
        self.launch_mask = launch_mask
        self.x0 = x0
        self.revenue_model = revenue_model
        self.econ_params = econ_params
        self.lam = lam
        self.fringe_start_slice = fringe_start_slice
        self.fringe_end_slice = fringe_end_slice
        self.tspan = np.linspace(0, 1, 2)

        # This is the number of all objects in each shell. Starts as x0 (initial population)
        self.current_environment = x0 

    def excess_return_calculator(self, launches):
        """
            Calculate the excess return for the given state matrix and launch rates.

            Launches: Open-access launch rates. This is just the fringe satellites. 1 x n_shells. 
        """
        
        # Calculate excess returns
        self.lam[self.fringe_start_slice:self.fringe_end_slice] = launches

        # fringe_launches = self.fringe_launches # This will be the first guess by the model 
        state_next_path = self.MOCAT.propagate(self.tspan, self.x0, self.lam)

        # gets the final output and update the current environment matrix
        state_next = state_next_path[-1, :]
        self.current_environment = state_next
        
        # ADD ADR STUFF HERE??????
        

        
        # Calculate the probability of collision based on the new positions
        collision_probability = self.calculate_probability_of_collision(state_next)

        rate_of_return = self.fringe_rate_of_return(state_next, collision_probability)

        # Calculate the excess rate of return
        excess_returns = (rate_of_return - collision_probability*(1 + self.econ_params.tax)) #*100

        return excess_returns

    def calculate_probability_of_collision(self, state_matrix):
        """
            In the MOCAT Configuration, the indicated for active loss probability is already created. Now in the code, you just need to pass the state 
            matrix.

            Return: 
                - Active Loss per shell. This can be used to infer collision probability.  
        """
        evaluated_value = self.MOCAT.scenario_properties.fringe_active_loss(*state_matrix)
        evaluated_value_flat = [float(value[0]) for value in evaluated_value]
        return np.array(evaluated_value_flat)
    
    def fringe_rate_of_return(self, state_matrix, collision_risk):

        if self.revenue_model == "linear":           
            fringe_total = state_matrix[self.fringe_start_slice:self.fringe_end_slice]

            revenue = self.econ_params.intercept - self.econ_params.coef * np.sum(fringe_total)
            revenue = revenue * self.launch_mask

            discount_rate = self.econ_params.discount_rate

            depreciation_rate = 1 / self.econ_params.sat_lifetime

            # Equilibrium expression for rate of return.
            rev_cost = revenue / self.econ_params.cost

            if self.econ_params.bond is None:
                rate_of_return = rev_cost - discount_rate - depreciation_rate  
            else:
                bond_per_shell = collision_risk * self.econ_params.bond
                bond = ((1-self.econ_params.comp_rate) * (bond_per_shell / self.econ_params.cost))
                rate_of_return = rev_cost - discount_rate - depreciation_rate - bond

            # print("Rate of return",  rate_of_return)
        else:
            # Other revenue models can be implemented here
            rate_of_return = 0 

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

        # Apply the launch mask to the initial guess
        launch_rate_init = self.solver_guess * self.launch_mask
        print(sum(launch_rate_init))

        # Define bounds for  the solver
        lower_bound = np.zeros_like(launch_rate_init)  # Lower bound is zero

        # Define solver options
        solver_options = {
            'method': 'trf',  #  Trust Region Reflective algorithm = trf
            'verbose': 0  # Show output if not parallelized
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

        print("i")

        return launch_rate
    
def create_bar_chart(data, title, xlabel, ylabel, filename):
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(data)), data, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(len(data)))
    plt.tight_layout()
    plt.savefig(f"figures/{filename}")
    plt.close()
    print(f"{title} bar chart saved as {filename}")
