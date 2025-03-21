import numpy as np

def evaluate_pmd(state_matrix, pmd_rate, deltat, fringe_start_slice, fringe_end_slice,
                 derelict_start_slice, derelict_end_slice, econ_parms):
    """
    For OPUS, the PMD rate is variable. Here we adjust the state matrix based on the 
    PMD rate. In particular, derelicts computed in shells that are non-compliant 
    (naturally_compliant_vector == 0) are summed and added to the last compliant shell.
    
    Parameters
    ----------
    state_matrix : np.array
        Array representing the number of satellites in each shell.
    pmd_rate : float
        The PMD rate used to compute derelict generation.
    deltat : float
        The active lifetime (or time step duration) used in calculations.
    fringe_start_slice, fringe_end_slice : slice indices
        The indices defining the "fringe" region where satellites are reduced.
    derelict_start_slice, derelict_end_slice : slice indices
        The indices defining where derelict satellites are stored.
    econ_parms : object
        An object that has an attribute `naturally_compliant_vector`, a NumPy array where
        a value of 1 indicates compliance and 0 indicates non-compliance.
        
    Returns
    -------
    state_matrix : np.array
        Updated state matrix after applying PMD adjustments.
    """

    # Get the number of items in each shell within the fringe range.
    num_items_fringe = state_matrix[fringe_start_slice:fringe_end_slice]

    # Compute the number of derelicts for each shell.
    compliant_derelicts = (1 / deltat) * num_items_fringe * pmd_rate

    # Non Compliant Derelicts
    non_compliant_derelicts = (1 / deltat) * num_items_fringe * (1-pmd_rate)
    
    # Remove the appropriate number of fringe satellites based on active lifetime.
    state_matrix[fringe_start_slice:fringe_end_slice] -= (1 / deltat) * num_items_fringe


    # There are three posibilites. 
    # 1. The shell is naturally compliant, so all derelicts remain where they are. 

    # 2. The compliant_derelicts are added to the last compliant shell.

    # 3. The non_compliant_derelicts remain where they are. 
    
    # Find the index of the last naturally compliant shell.
    last_compliant_shell = np.where(econ_parms.naturally_compliant_vector == 1)[0][-1]

    # Identify non-compliant shells
    non_compliant_mask = (econ_parms.naturally_compliant_vector == 0)

    # Sum the derelict numbers for non-compliant shells.
    sum_non_compliant = np.sum(compliant_derelicts[non_compliant_mask])

    # Add the sum of non-compliant derelicts to the last compliant shell.
    compliant_derelicts[last_compliant_shell] += sum_non_compliant

    # Zero out the derelict counts in the non-compliant shells.
    compliant_derelicts[non_compliant_mask] = 0

    # Now add the adjusted derelict numbers to the state matrix in the derelict slice.
    state_matrix[derelict_start_slice:derelict_end_slice] += compliant_derelicts

    # Now for the non_compliant derelicts. They just remain where they are. 
    state_matrix[derelict_start_slice:derelict_end_slice] += non_compliant_derelicts
 
    return state_matrix