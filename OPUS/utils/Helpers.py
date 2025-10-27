def insert_launches_into_lam(lam_vector, launches, multi_species, elliptical):
    """
    Insert flattened launch values into the appropriate slices of the lam_vector,
    based on species start/end indices.

    This is specifically for multi_species, when the index of the species does not fit
    directly into the pyssem launch array. 

    Parameters:
        lam_vector (np.ndarray): The full lambda vector to modify (in-place).
        launches (np.ndarray): Flat array of launches, ordered by species.
        multi_species (object): An object with a .species list, where each species
                                has .start_slice and .end_slice attributes.
    """

    pointer = 0
    for species in multi_species.species:
        start_slice = species.start_slice
        end_slice = species.end_slice
        n_shells = end_slice - start_slice

        species_launch = launches[pointer:pointer + n_shells]
        
        if elliptical:
            # launches are now in a 3d matrix
            lam_vector[:, species.species_idx, 0] = species_launch
        else:
            lam_vector[start_slice:end_slice] = species_launch

        pointer += n_shells

    return lam_vector