import numpy as np
from scipy.optimize import least_squares
import sympy as sp
from concurrent.futures import ProcessPoolExecutor
from pyssem.model import Model
import matplotlib.pyplot as plt

class OpenAccessSolver:
    def __init__(self, MOCAT: Model, solver_guess, launch_mask, x0, revenue_model, 
                 econ_params, lam, n_workers, fringe_start_slice, fringe_end_slice):
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

        total = 0
        for i in lam:
            if i is not None:
                total += sum(i)
        
        print("Total launches: ", total)

        total = 0
        for i in lam:
            if i is not None:
                total += sum(i)
        
        print("Total launches after: ", total)

        # fringe_launches = self.fringe_launches # This will be the first guess by the model 
        state_next_path = self.MOCAT.propagate(self.tspan, self.x0, lam)

        # gets the final output and update the current environment matrix
        state_next = state_next_path[-1, :]
        self.current_environment = state_next

        difference = state_next - self.x0

        # Split the difference into 4 species: S, Su, N, D
        # Extract values from lam, replacing None with 0
        lam_values = [item[0] if item is not None and item[0] is not None else 0 for item in lam]

        # Split the difference and lam into 4 species: S, Su, N, D
        species = ["S", "Su", "N", "D"]
        difference_species_data = [difference[i * 10:(i + 1) * 10] for i in range(4)]
        lam_species_data = [lam_values[i * 10:(i + 1) * 10] for i in range(4)]

        # # Create a 2x2 subplot for the 4 species
        # fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # for i, ax in enumerate(axes.flat):
        #     # Plot the difference as bars
        #     ax.bar(range(1, 11), difference_species_data[i], label="Difference", alpha=0.7)
        #     # Plot lam as a line
        #     ax.plot(range(1, 11), lam_species_data[i], color='red', label="Lam Values", linestyle='--', marker='o')
        #     ax.set_title(f"Difference and Lam Plot for {species[i]}")
        #     ax.set_xlabel("Index")
        #     ax.set_ylabel("Value")
        #     ax.set_xticks(range(1, 11, 5))
        #     ax.grid(axis='y', linestyle='--', alpha=0.7)
        #     ax.legend()

        # plt.tight_layout()
        # plt.savefig("propagation_difference.png")

        # Calculate the probability of collision based on the new positions
        collision_probability = self.calculate_probability_of_collision(state_next)

        # Testing
        collision_probability = collision_probability * 107012

        rate_of_return = self.fringe_rate_of_return(state_next)

        # Calculate the excess rate of return
        excess_returns = 100 * (rate_of_return - collision_probability*(1 + self.econ_params.tax))

        # create bar chart of excess returns and savefig

        # plt.figure(figsize=(10, 6))
        # plt.bar(range(len(excess_returns)), excess_returns, tick_label=[f"Value {i+1}" for i in range(len(excess_returns))])
        # plt.xlabel("Index")
        # plt.ylabel("Value")
        # plt.title("Bar Plot of Excess Returns")
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.savefig("excess_returns.png")

        return excess_returns

    def calculate_probability_of_collision(self, state_matrix):
        
        # THIS WILL BE IN THE MOCAT MODEL - NOT HERE
        # Currently the model is calcualting the probability of collision for each shell. I think this can be done for all shells at once. 
        # x0 is the current state of the system.

        evaluated_value = self.MOCAT.scenario_properties.fringe_active_loss(*state_matrix)
        evaluated_value_flat = [float(value[0]) for value in evaluated_value]

        # 1 - exp(evaluated_values_flat) # Probability of collision?

        # # Create a bar plot for `evaluated_value`
        # plt.figure(figsize=(10, 6))
        # plt.bar(range(len(evaluated_value_flat)), evaluated_value_flat, tick_label=[f"Value {i+1}" for i in range(len(evaluated_value_flat))])
        # plt.xlabel("Index")
        # plt.ylabel("Value")
        # plt.title("Bar Plot of Evaluated Values")
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.savefig("collision_probability.png")
        # plt.show()

        return np.array(evaluated_value_flat)
    
    def fringe_rate_of_return(self, state_matrix):

        if self.revenue_model == "linear":
            # Linear revenue model: intercept - coef * total_number_of_fringe_satellites
            
            fringe_total = state_matrix[self.fringe_start_slice:self.fringe_end_slice]

            # print("Sum of sats: ", sum(fringe_total))
            revenue = self.econ_params.intercept - self.econ_params.coef * np.sum(fringe_total)
            revenue = revenue * self.launch_mask
            # print("Revenue: ",  sum(revenue))
            cost = self.econ_params.cost
            # print("Cost: ", cost)
            
            discount_rate = self.econ_params.discount_rate

            depreciation_rate = 1 / self.econ_params.sat_lifetime

            #revenue = np.array([435620] * self.MOCAT.scenario_properties.n_shells)
            rev_cost = revenue / cost
            rate_of_return = rev_cost - discount_rate - depreciation_rate  # Equilibrium expression for rate of return.

            # print("Rate of return",  rate_of_return)
        else:
            # Other revenue models can be implemented here
            rate_of_return = 0  # Placeholder value

        return rate_of_return
    
    def solve(self):
        return self.excess_return_calculator(self.state_matrix, self.lam)

        
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
    
    def find_initial_guess(self):
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

        # Propagate the model
        state_next_path = self.MOCAT.propagate(self.tspan, self.current_environment, self.lam)

        # Get the final output and update the current environment matrix
        state_next = state_next_path[-1, :]
        self.current_environment = state_next

        # Calculate the probability of collision based on the new positions
        collision_probability = self.calculate_probability_of_collision(state_next)

        # Calculate the rate of return
        rate_of_return = self.fringe_rate_of_return(state_next)

        # Calculate the excess rate of return
        excess_returns = 100 * (rate_of_return - collision_probability*(1 + self.econ_params.tax))

        # Initial guess of fringe satellites
        fringe_initial_guess = np.zeros_like(self.x0)
        fringe_initial_guess = excess_returns * self.launch_mask

        return fringe_initial_guess 



