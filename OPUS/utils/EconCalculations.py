import numpy as np

class EconCalculations:
    """
    A class to encapsulate and manage economic calculations within the simulation loop.
    
    This class holds the state of economic variables that persist between time periods.
    """
    def __init__(self, econ_params, initial_removal_cost=5000000.0):
        """
        Initializes the economic calculator.
        
        Args:
            econ_params: An object containing economic parameters, like the welfare coefficient.
            initial_removal_cost (float): The cost to remove one piece of debris.
        """
        # --- Parameters ---
        self.welfare_coef = econ_params.coef
        self.removal_cost = initial_removal_cost

        # --- State Variable ---
        # available for removals at the start of a period (previous revenue + all rollovers).
        self.total_funds_for_removals = 0.0

    def get_removals_for_current_period(self):
        """
        Calculates how many removals can be afforded in the current period
        based on the *total available funds*.
        
        Returns:
            int: The number of affordable removals.
        """
        if self.removal_cost > 0:
            return int(self.total_funds_for_removals // self.removal_cost)
        return 0

    def process_period_economics(self, num_actually_removed, current_environment, fringe_slices, new_tax_revenue):
        """
        Performs all economic calculations for the current period and updates the state for the next.
        
        Args:
            num_actually_removed (int): The number of objects removed in this period.
            current_environment (np.ndarray): The state of the orbital environment.
            fringe_slices (tuple): A tuple (start, end) for the fringe satellite slice.
            new_tax_revenue (float): The total tax revenue generated in this period.
            
        Returns:
            tuple: A tuple containing (welfare, funds_left_before_new_revenue).
        """
        fringe_start_slice, fringe_end_slice = fringe_slices

        # 1. Calculate the cost of this period's removals and the remaining funds
        cost_of_removals = num_actually_removed * self.removal_cost
        funds_left_before_new_revenue = self.total_funds_for_removals - cost_of_removals
        
        # 2. Calculate welfare for this period.
        # The welfare bonus is based on the unspent funds from the available budget.
        welfare_revenue_component = max(0, funds_left_before_new_revenue)
        
        total_fringe_sat = np.sum(current_environment[fringe_start_slice:fringe_end_slice])
        welfare = 0.5 * self.welfare_coef * total_fringe_sat**2 + welfare_revenue_component
        
        # 3. Update the state for the NEXT period.
        # The new pool of funds is the leftover amount plus the newly collected tax revenue.
        self.total_funds_for_removals = funds_left_before_new_revenue + new_tax_revenue
        
        return welfare, funds_left_before_new_revenue
    
def revenue_open_access_calculations(open_access_inputs, state_next):
    collision_probability = open_access_inputs._last_collision_probability
    #J ─── Revenue collection ────────────────────────────────────────────────────
    fringe_total = state_next[open_access_inputs.fringe_start_slice:open_access_inputs.fringe_end_slice]

    if (open_access_inputs.econ_params.bond is not None) and (open_access_inputs.econ_params.bond != 0):
        # vector already created inside fringe_rate_of_return()
        revenue_by_shell = open_access_inputs.bond_revenue                      # bond
        open_access_inputs._revenue_type = "bond"

    elif getattr(open_access_inputs.econ_params, "ouf", 0) != 0:
        revenue_by_shell = open_access_inputs.econ_params.ouf * collision_probability * fringe_total    # OUF
        open_access_inputs._revenue_type = "ouf"

    elif open_access_inputs.econ_params.tax != 0: #tax
        Cp            = collision_probability
        cost_per_sat  = np.asarray(open_access_inputs.econ_params.cost)
        revenue_by_shell = open_access_inputs.econ_params.tax * Cp * cost_per_sat * fringe_total
        open_access_inputs._revenue_type = "tax"

    else:                                                      # nothing levied
        revenue_by_shell = np.zeros_like(fringe_total)
        open_access_inputs._revenue_type = "none"

    total_revenue = revenue_by_shell.sum()

    _last_tax_revenue   = revenue_by_shell
    _last_total_revenue = float(total_revenue)
    _dbg_tax_rate       = open_access_inputs.econ_params.tax
    _dbg_Cp             = collision_probability  
    _dbg_cost_per_sat   = np.asarray(open_access_inputs.econ_params.cost)
    _dbg_fringe_total   = fringe_total

    return _last_tax_revenue, _last_total_revenue, _dbg_tax_rate, _dbg_Cp, _dbg_cost_per_sat, _dbg_fringe_total