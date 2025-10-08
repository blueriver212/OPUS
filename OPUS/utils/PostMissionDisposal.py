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
            successful_pmd = 0.97 * (1 / species.deltat) * num_items_fringe
            failed_pmd = 0.03 * (1 / species.deltat) * num_items_fringe

            # Remove all satellites at end of life
            state_matrix[start:end] -= (1 / species.deltat) * num_items_fringe

            # Get naturally compliant shell indices
            compliant_indices = np.where(species.econ_params.naturally_compliant_vector == 1)[0]

            # Distribute failed PMD randomly across compliant shells
            derelict_addition = np.zeros_like(num_items_fringe)

            total_failed = np.sum(failed_pmd)
            n_compliant = len(compliant_indices)

            # add all failed satellites to the last compliant shell
            derelict_addition[compliant_indices[-1]] = total_failed

            # if n_compliant > 0:
            #     # Generate a multinomial draw to split total_failed across compliant shells
            #     proportions = np.random.multinomial(int(round(total_failed)), [1/n_compliant] * n_compliant)

            #     for idx, shell_idx in enumerate(compliant_indices):
            #         derelict_addition[shell_idx] += proportions[idx]

            state_matrix[derelict_start:derelict_end] += derelict_addition

            species.sum_compliant = np.sum(successful_pmd)
            species.sum_non_compliant = np.sum(failed_pmd)

        elif species_name == 'Su':
            # All compliant PMD are dropped at the highest naturall compliant vector. 
            last_compliant_shell = np.where(species.econ_params.naturally_compliant_vector == 1)[0][-1]
            non_compliant_mask = (species.econ_params.naturally_compliant_vector == 0)

            compliant_derelicts = (1 / species.deltat) * num_items_fringe * species.econ_params.pmd_rate
            non_compliant_derelicts = (1 / species.deltat) * num_items_fringe * (1 - species.econ_params.pmd_rate)

            state_matrix[start:end] -= (1 / species.deltat) * num_items_fringe

            sum_non_compliant = np.sum(compliant_derelicts[non_compliant_mask])
            compliant_derelicts[last_compliant_shell] += sum_non_compliant
            compliant_derelicts[non_compliant_mask] = 0

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
 

def evaluate_pmd_elliptical(state_matrix, multi_species):
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
            S_matrix = state_matrix[:, species.species_idx, 0]

            # 97% removed from sim, 3% fail PMD and get dropped randomly into compliant shells
            successful_pmd = 0.97 * (1 / species.deltat) * S_matrix
            failed_pmd = 0.03 * (1 / species.deltat) * S_matrix

            # Remove all satellites at end of life
            state_matrix[:, species.species_idx, 0] -= (1 / species.deltat) * S_matrix

            # Get naturally compliant shell indices
            compliant_indices = np.where(species.econ_params.naturally_compliant_vector == 1)[0]

            # Distribute failed PMD randomly across compliant shells
            derelict_addition = np.zeros_like(S_matrix)

            total_failed = np.sum(failed_pmd)
            n_compliant = len(compliant_indices)
            
            # Debug information
            # print(f"Species {species_name}: total_failed={total_failed:.6f}, n_compliant={n_compliant}")
            # print(f"  compliant_indices: {compliant_indices}")
            # print(f"  naturally_compliant_vector: {species.econ_params.naturally_compliant_vector}")

            # if n_compliant > 0:
            #     if n_compliant == 1:
            #         # If only one compliant shell, assign all failed satellites to it directly
            #         derelict_addition[compliant_indices[0]] = total_failed
            #     else:
            #         # Generate a multinomial draw to split total_failed across compliant shells
            #         proportions = np.random.multinomial(int(round(total_failed)), [1/n_compliant] * n_compliant)

            #         for idx, shell_idx in enumerate(compliant_indices):
            #             derelict_addition[shell_idx] += proportions[idx]
                
            #     # Check if sum of derelict_addition equals total_failed
            #     derelict_sum = np.sum(derelict_addition)
            #     if abs(derelict_sum - total_failed) > 1e-10:  # Allow for small floating point differences
            #         print(f"WARNING: derelict_addition sum ({derelict_sum}) != total_failed ({total_failed})")
            #         print(f"  n_compliant: {n_compliant}, compliant_indices: {compliant_indices}")
            #         print(f"  proportions: {proportions}")
            #         print(f"  derelict_addition: {derelict_addition}")

            # add all failed satellites to the last compliant shell
            derelict_addition[compliant_indices[-1]] = total_failed

            state_matrix[:, species.derelict_idx, 0] += derelict_addition

            species.sum_compliant = np.sum(successful_pmd)
            species.sum_non_compliant = np.sum(failed_pmd)

        elif species_name == 'Su':
            # get Su matrix
            Su_matrix = state_matrix[:, species.species_idx, 0]
            
            # All compliant PMD are dropped at the highest naturally compliant vector
            last_compliant_shell = np.where(species.econ_params.naturally_compliant_vector == 1)[0][-1]
            non_compliant_mask = (species.econ_params.naturally_compliant_vector == 0)

            compliant_derelicts = (1 / species.deltat) * Su_matrix * species.econ_params.pmd_rate
            non_compliant_derelicts = (1 / species.deltat) * Su_matrix * (1 - species.econ_params.pmd_rate)

            # Remove all satellites at end of life
            state_matrix[:, species.species_idx, 0] -= (1 / species.deltat) * Su_matrix

            sum_non_compliant = np.sum(compliant_derelicts[non_compliant_mask])
            compliant_derelicts[last_compliant_shell] += sum_non_compliant
            compliant_derelicts[non_compliant_mask] = 0

            state_matrix[:, species.derelict_idx, 0] += compliant_derelicts
            state_matrix[:, species.derelict_idx, 0] += non_compliant_derelicts

            species.sum_compliant = np.sum(compliant_derelicts)
            species.sum_non_compliant = np.sum(non_compliant_derelicts)

        elif species_name == 'Sns':
            # get Sns matrix
            Sns_matrix = state_matrix[:, species.species_idx, 0]
            
            # No PMD; everything goes to derelict in place
            derelicts = (1 / species.deltat) * Sns_matrix

            state_matrix[:, species.species_idx, 0] -= derelicts
            state_matrix[:, species.derelict_idx, 0] += derelicts

            species.sum_compliant = 0
            species.sum_non_compliant = np.sum(derelicts)

        else:
            raise ValueError(f"Unhandled species type: {species_name}")

    return state_matrix, multi_species


