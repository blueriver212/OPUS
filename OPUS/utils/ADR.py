import numpy as np

# loosely based on PMD function
def implement_adr(state_matrix,MOCAT,adr_params):
    if len(adr_params.target_species) == 0:
        state_matrix = state_matrix
    else:
        for i, sp in enumerate(MOCAT.scenario_properties.species_names):
            params = adr_params.properties.get(sp) 
            start = i*MOCAT.scenario_properties.n_shells
            end = (i+1)*MOCAT.scenario_properties.n_shells

            max_indices = []
            if params.target == 1:
                sorted_mat = state_matrix[start:end].sort(reverse=True)
                for j in (params.n_max - 1):
                    max_indices[j] = state_matrix[start:end].index(sorted_mat[j])
                    state_matrix[start:end][max_indices[j]] *= (1-params.p_remove)
                # for j in params.target_shell:
                #     state_matrix[start:end][j-1] *= (1-params.p_remove)

        return state_matrix