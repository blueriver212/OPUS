import numpy as np

# def evaluate_pmd(state_matrix, multi_species):
#     """
#     NOW FOR MULTI SPECIES
#     """
#     # State matrix holds all the entire environment, multi_species holds all of the information required for PMD

#     for species in multi_species.species:
#         # FIND COMPLIANT SHELLS
#         # Find he idx of the last naturally compliant shell 
#         last_compliant_shell = np.where(species.econ_params.naturally_compliant_vector == 1)[0][-1]

#         # Identify non-compliant shells
#         non_compliant_mask = (species.econ_params.naturally_compliant_vector == 0)

#         # Get the number of items in each shell within the fringe range
#         num_items_fringe = state_matrix[species.start_slice:species.end_slice]

#         # Compute the number of derelicts for each shell.
#         # If your pmd_rate is 65% (0.65) then your compliance is 0.65 of your annual amount. 
#         compliant_derelicts = (1 / species.deltat) * num_items_fringe * species.econ_params.pmd_rate

#         # Non compliance derelicts
#         # Then your non-compliance is 1-0.63 = 0.35 of your annnual ammount
#         non_compliant_derelicts = (1 / species.deltat) * num_items_fringe * (1-species.econ_params.pmd_rate)

#         # Remove the appropriate number of fringe satellites based on active lifetime.
#         state_matrix[species.start_slice:species.end_slice] -= (1 / species.deltat) * num_items_fringe

#         # There are three posibilites. 
#         # 1. The shell is naturally compliant, so all derelicts remain where they are. 

#         # 2. The compliant_derelicts are added to the last compliant shell.

#         # 3. The non_compliant_derelicts remain where they are. 

#         # Sum the derelict numbers for non-compliant shells.
#         sum_non_compliant = np.sum(compliant_derelicts[non_compliant_mask])

#         # Add the sum of non-compliant derelicts to the last compliant shell.
#         compliant_derelicts[last_compliant_shell] += sum_non_compliant

#         # Zero out the derelict counts in the non-compliant shells.
#         compliant_derelicts[non_compliant_mask] = 0

#         # Now add the adjusted derelict numbers to the state matrix in the derelict slice.
#         state_matrix[species.derelict_start_slice:species.derelict_end_slice] += compliant_derelicts

#         # Now for the non_compliant derelicts. They just remain where they are. 
#         state_matrix[species.derelict_start_slice:species.derelict_end_slice] += non_compliant_derelicts

#         # Save required information to the species
#         species.sum_non_compliant = sum_non_compliant
#         species.sum_compliant = sum(compliant_derelicts)
    
#     return state_matrix, multi_species

def evaluate_pmd(state_matrix, multi_species):
    """
    PMD logic applied per species type:
    - 'S': 97% removed, 3% go to top compliant shell
    - 'Su': current logic (PMD-compliant to last compliant shell, non-compliant stay in shell)
    - 'Sns': no PMD, all become derelicts in place
    """
    
    for species in multi_species.species:
        species_name = species.name
        start = species.start_slice
        end = species.end_slice
        derelict_start = species.derelict_start_slice
        derelict_end = species.derelict_end_slice

        num_items_fringe = state_matrix[start:end]

        if species_name == 'S':
            # 97% removed from sim, 3% fail PMD and get dropped randomly into compliant shells
            successful_pmd = species.econ_params.pmd_rate * (1 / species.deltat) * num_items_fringe
            failed_pmd = (1 - species.econ_params.pmd_rate) * (1 / species.deltat) * num_items_fringe

             # Remove all satellites at end of life
            state_matrix[start:end] -= (1 / species.deltat) * num_items_fringe

            # Get naturally compliant shell indices
            compliant_indices = np.where(species.econ_params.naturally_compliant_vector == 1)[0]

            # Distribute failed PMD to highest compliant shell only
            derelict_addition = np.zeros_like(num_items_fringe)

            total_failed = np.sum(failed_pmd)
            
            if len(compliant_indices) > 0:
                # Find the highest compliant shell (assuming higher index = higher altitude)
                highest_compliant_shell = np.max(compliant_indices)
                
                # Place all failed PMD satellites in the highest compliant shell
                derelict_addition[highest_compliant_shell] = total_failed

            state_matrix[derelict_start:derelict_end] += derelict_addition

            species.sum_compliant = np.sum(successful_pmd)
            species.sum_non_compliant = np.sum(failed_pmd)

        elif species_name == 'Su':
            # All compliant PMD are dropped at the highest naturall compliant vector. 
            last_compliant_shell = np.where(species.econ_params.naturally_compliant_vector == 1)[0][-1]
            non_compliant_mask = (species.econ_params.naturally_compliant_vector == 0)

            # calculate compliant and non-compliant derelicts - just for reporting
            compliant_derelicts = (1 / species.deltat) * num_items_fringe * species.econ_params.pmd_rate
            non_compliant_derelicts = (1 / species.deltat) * num_items_fringe * (1 - species.econ_params.pmd_rate)

            # remove all satellites at end of life
            state_matrix[start:end] -= (1 / species.deltat) * num_items_fringe

            # add compliant derelicts to last compliant shell
            sum_non_compliant = np.sum(compliant_derelicts[non_compliant_mask])
            compliant_derelicts[last_compliant_shell] += sum_non_compliant
            compliant_derelicts[non_compliant_mask] = 0

            # add compliant and non-compliant derelicts to derelict slice
            state_matrix[derelict_start:derelict_end] += compliant_derelicts
            state_matrix[derelict_start:derelict_end] += non_compliant_derelicts

            species.sum_compliant = np.sum(compliant_derelicts)
            species.sum_non_compliant = np.sum(non_compliant_derelicts)

        elif species_name == 'Sns':
            # No PMD; everything goes to derelict in place
            derelicts = (1 / species.deltat) * num_items_fringe

            state_matrix[start:end] -= derelicts
            state_matrix[derelict_start:derelict_end] += derelicts

            species.sum_compliant = 0
            species.sum_non_compliant = np.sum(derelicts)

        else:
            raise ValueError(f"Unhandled species type: {species_name}")

    return state_matrix, multi_species
 

def evaluate_pmd_elliptical(state_matrix, state_matrix_alt, multi_species):
    """
    PMD logic applied per species type:
    - 'S': 97% removed, 3% go to top compliant shell
    - 'Su': current logic (PMD-compliant to last compliant shell, non-compliant stay in shell)
    - 'Sns': no PMD, all become derelicts in place
    """
    
    for species in multi_species.species:
        species_name = species.name
        
        if species_name == 'S':
            # get S matrix
            num_items_fringe = state_matrix[:, species.species_idx, 0]

            # 97% removed from sim, 3% fail PMD and get dropped randomly into compliant shells
            successful_pmd = species.econ_params.pmd_rate * (1 / species.deltat) * num_items_fringe
            failed_pmd = (1 - species.econ_params.pmd_rate) * (1 / species.deltat) * num_items_fringe

             # Remove all satellites at end of life - from both sma and alt bins
            state_matrix[:, species.species_idx, 0] -= (1 / species.deltat) * num_items_fringe
            state_matrix_alt[:, species.species_idx] -= (1 / species.deltat) * num_items_fringe

            # Get naturally compliant shell indices
            compliant_indices = np.where(species.econ_params.naturally_compliant_vector == 1)[0]

            # Distribute failed PMD to highest compliant shell only
            derelict_addition = np.zeros_like(num_items_fringe)

            total_failed = np.sum(failed_pmd)
            
            if len(compliant_indices) > 0:
                # Find the highest compliant shell (assuming higher index = higher altitude)
                highest_compliant_shell = np.max(compliant_indices)
                
                # Place all failed PMD satellites in the highest compliant shell
                derelict_addition[highest_compliant_shell] = total_failed

            state_matrix[:, species.derelict_idx, 0] += derelict_addition
            state_matrix_alt[:, species.derelict_idx] += derelict_addition
            
            species.sum_compliant = np.sum(successful_pmd)
            species.sum_non_compliant = np.sum(failed_pmd)

        elif species_name == 'Su':
            # get Su matrix
            num_items_fringe = state_matrix[:, species.species_idx, 0]
            
            # All compliant PMD are dropped at the highest naturall compliant vector. 
            last_compliant_shell = np.where(species.econ_params.naturally_compliant_vector == 1)[0][-1]
            non_compliant_mask = (species.econ_params.naturally_compliant_vector == 0)

            # Number of compliant and non compiant satellites in each cell 
            compliant_derelicts = (1 / species.deltat) * num_items_fringe * species.econ_params.pmd_rate
            non_compliant_derelicts = (1 / species.deltat) * num_items_fringe * (1 - species.econ_params.pmd_rate)

            # remove all satellites at end of life
            state_matrix[:, species.species_idx, 0] -= (1 / species.deltat) * num_items_fringe
            state_matrix_alt[:, species.species_idx] -= (1 / species.deltat) * num_items_fringe

            # add compliant derelicts to last compliant shell
            sum_non_compliant = np.sum(compliant_derelicts[non_compliant_mask])
            compliant_derelicts[last_compliant_shell] += sum_non_compliant
            compliant_derelicts[non_compliant_mask] = 0

            # this should be the compliant going to the top of the PMD lifetime shell and the non compliant remaining in the same shell
            derelict_addition = non_compliant_derelicts + compliant_derelicts

            # add derelicts back to both sma and altitude matrices
            state_matrix[:, species.derelict_idx, 0] += derelict_addition
            state_matrix_alt[:, species.derelict_idx] += derelict_addition

            species.sum_compliant = np.sum(compliant_derelicts)
            species.sum_non_compliant = np.sum(non_compliant_derelicts)

        elif species_name == 'Sns':
            # get Sns matrix
            num_items_fringe = state_matrix[:, species.species_idx, 0]

            derelicts = (1 / species.deltat) * num_items_fringe

            state_matrix[:, species.species_idx, 0] -= derelicts
            state_matrix[:, species.derelict_idx, 0] += derelicts

            state_matrix_alt[:, species.species_idx] -= derelicts
            state_matrix_alt[:, species.derelict_idx] += derelicts

            species.sum_compliant = 0
            species.sum_non_compliant = np.sum(derelicts)

        else:
            raise ValueError(f"Unhandled species type: {species_name}")

    return state_matrix, state_matrix_alt, multi_species


