import numpy as np

# loosely based on PMD function
def implement_adr(state_matrix,MOCAT,adr_params):
    if len(adr_params.target_species) == 0:
        state_matrix = state_matrix
    elif adr_params.target_species is not None:
        for i, sp in enumerate(MOCAT.scenario_properties.species_names):
            params = adr_params.properties.get(sp) 
            start = i*MOCAT.scenario_properties.n_shells
            end = (i+1)*MOCAT.scenario_properties.n_shells
            old = state_matrix[start:end]
            max_indices = [0] * params.get("n_max")
            if params.get("target") == 1:
                # sorted_mat = sorted(state_matrix[start:end], reverse=True)
                # for j in range(len(max_indices)):
                #     jj = j-1
                #     max_indices[jj] = np.where(state_matrix[start:end]==sorted_mat[jj])
                #     state_matrix[start:end][max_indices[jj]] *= (1-params.get("p_remove"))
                for j in params.get("target_shell"):
                    state_matrix[start:end][j-1] *= (1-params.get("p_remove"))
            else:
                state_matrix = state_matrix
            diff = state_matrix[start:end]-old
            if any(x != 0 for x in diff):
                print("This is what's up with the ADR state matrix bit: " + str(diff))
    # print(state_matrix)

    return state_matrix