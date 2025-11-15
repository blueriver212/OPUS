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
            econ_params: An object containing economic parameters (used for defaults, like coef).
            initial_removal_cost (float): The cost to remove one piece of debris.
        """
        # --- Parameters ---
        # We store the *default* welfare_coef here, but will use species-specific ones in the loop.
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

    def process_period_economics(self, num_actually_removed, current_environment, multi_species, new_tax_revenue):
        """
        Performs all economic calculations for the current period and updates the state for the next.
        
        Args:
            num_actually_removed (int): The number of objects removed in this period.
            current_environment (np.ndarray): The state of the orbital environment.
            multi_species (MultiSpecies): The object containing all species objects.
            new_tax_revenue (float): The total tax revenue generated in this period.
            
        Returns:
            tuple: A tuple containing (welfare, funds_left_before_new_revenue).
        """
        # 1. Calculate the cost of this period's removals and the remaining funds
        cost_of_removals = num_actually_removed * self.removal_cost
        funds_left_before_new_revenue = self.total_funds_for_removals - cost_of_removals
        
        # 2. Calculate welfare for this period.
        # The welfare bonus is based on the unspent funds from the available budget.
        welfare_revenue_component = max(0, funds_left_before_new_revenue)
        
        welfare_satellite_component = 0.0
        
        # Determine if the environment is elliptical or circular
        is_elliptical = current_environment.ndim > 1 

        for species in multi_species.species:
            # Only calculate satellite welfare for active species (S* and Su*)
            if species.name.startswith('S'):
                species_coef = species.econ_params.coef
                
                if is_elliptical:
                    # Sum across all shells and eccentricity bins (assuming [shell, species_idx, ecc_bin])
                    total_sats = np.sum(current_environment[:, species.species_idx, :])
                else:
                    # Sum across all shells for this species
                    total_sats = np.sum(current_environment[species.start_slice:species.end_slice])
                
                welfare_satellite_component += 0.5 * species_coef * total_sats**2
        
        welfare = welfare_satellite_component + welfare_revenue_component
        
        # 3. Update the state for the NEXT period.
        # The new pool of funds is the leftover amount plus the newly collected tax revenue.
        self.total_funds_for_removals = funds_left_before_new_revenue + new_tax_revenue
        
        return welfare, funds_left_before_new_revenue
    
def revenue_open_access_calculations(open_access_inputs, state_next):
    
    
    # Get collision probabilities for ALL species (this is a dict)
    collision_probability_dict = open_access_inputs._last_collision_probability

    # Get the number of shells from the first species (all are the same)
    n_shells = open_access_inputs.multi_species.species[0].econ_params.cost.shape[0]
    
    # Initialize arrays to sum revenues
    total_revenue_by_shell = np.zeros(n_shells)
    total_bond_revenue_by_shell = np.zeros(n_shells)
    
    # Determine if the environment is elliptical or circular
    is_elliptical = state_next.ndim > 1
    
    # Store debug info (will be overwritten by last species, but that's ok for debugging)
    _dbg_tax_rate = 0
    _dbg_Cp = None
    _dbg_cost_per_sat = None
    _dbg_fringe_total = None

    for sp in open_access_inputs.multi_species.species:
        # Only calculate revenue for active species (S* and Su*)
        if not sp.name.startswith('S'):
            continue
            
        fringe_econ_params = sp.econ_params
        species_name = sp.name
        
        # Get the correct collision probability for THIS species
        if species_name not in collision_probability_dict:
            # This can happen for non-maneuverable species, default to 0
            collision_probability = np.zeros(n_shells)
        else:
            collision_probability = collision_probability_dict[species_name]

        # Get the satellite population for THIS species
        if is_elliptical:
            # Assuming state_next is [shell, species_idx] (altitude-only matrix from PMD)
            sats_total_per_shell = state_next[:, sp.species_idx]
        else:
            sats_total_per_shell = state_next[sp.start_slice:sp.end_slice]

        revenue_by_shell = np.zeros(n_shells)
        
        if (fringe_econ_params.bond is not None) and (fringe_econ_params.bond != 0):
            # Calculate revenue ONLY from sats at end-of-life
            sats_at_eol = sats_total_per_shell / fringe_econ_params.sat_lifetime
            
            # Calculate bond revenue: (non-compliance rate) * (sats at EOL) * (bond value)
            revenue_by_shell = (1 - fringe_econ_params.comp_rate) * sats_at_eol * fringe_econ_params.bond
            
            # Add to the running total for bond revenue
            total_bond_revenue_by_shell += revenue_by_shell
            open_access_inputs._revenue_type = "bond" # Mark that at least one species had a bond

        elif getattr(fringe_econ_params, "ouf", 0) != 0:
            revenue_by_shell = fringe_econ_params.ouf * collision_probability * sats_total_per_shell    # OUF
            open_access_inputs._revenue_type = "ouf"

        elif fringe_econ_params.tax != 0: #tax
            Cp = collision_probability
            cost_per_sat  = np.asarray(fringe_econ_params.cost)
            revenue_by_shell = fringe_econ_params.tax * Cp * cost_per_sat * sats_total_per_shell
            open_access_inputs._revenue_type = "tax"

        else:                                                      # nothing levied
            revenue_by_shell = np.zeros_like(sats_total_per_shell)
            if not hasattr(open_access_inputs, '_revenue_type'): # Don't overwrite if set by another species
                open_access_inputs._revenue_type = "none"

        # Add this species' revenue to the total
        total_revenue_by_shell += revenue_by_shell

        # Store debug info for this species
        _dbg_tax_rate       = fringe_econ_params.tax
        _dbg_Cp             = collision_probability  
        _dbg_cost_per_sat   = np.asarray(fringe_econ_params.cost)
        _dbg_fringe_total   = sats_total_per_shell

    # Set the solver's bond_revenue attribute to the grand total
    open_access_inputs.bond_revenue = total_bond_revenue_by_shell

    # Sum the total revenue across all shells
    total_revenue = total_revenue_by_shell.sum()

    _last_tax_revenue   = total_revenue_by_shell # This is now total revenue (tax, bond, or ouf)
    _last_total_revenue = float(total_revenue)
    
    # --- END OF UPDATE ---

    return _last_tax_revenue, _last_total_revenue, _dbg_tax_rate, _dbg_Cp, _dbg_cost_per_sat, _dbg_fringe_total