import numpy as np

# loosely based on PMD function
def implement_adr(state_matrix,MOCAT,adr_params):
    if len(adr_params.target_species) == 0:
        state_matrix = state_matrix
    elif adr_params.target_species is not None:
        if adr_params.time in adr_params.adr_times:
            for i, sp in enumerate(MOCAT.scenario_properties.species_names):
                params = adr_params.properties.get(sp) 
                if params is not None: 
                    start = i*MOCAT.scenario_properties.n_shells
                    end = (i+1)*MOCAT.scenario_properties.n_shells
                    old = state_matrix[start:end]
                    max_indices = [0] * params.get("n_max")
                    # to remove based on percentage:
                    if params.get("target") == 1:
                        # # targeting the shells with the most of the target species:
                        # sorted_mat = sorted(state_matrix[start:end], reverse=True)
                        # for j in range(len(max_indices)):
                        #     jj = j-1
                        #     max_indices[jj] = np.where(state_matrix[start:end]==sorted_mat[jj])
                        #     state_matrix[start:end][max_indices[jj]] *= (1-params.get("p_remove"))
                        
                        for j in params.get("target_shell"):
                            state_matrix[start:end][j-1] *= (1-params.get("p_remove"))
                else:
                    state_matrix = state_matrix

                # # to remove based on specific number:
                # if params.get("target") == 1:
                #     n_remove = params.get("n_remove")
                #     for j in params.get("target_shell"):
                #         if n_remove > state_matrix[start:end][j-1]:
                #             state_matrix[start:end][j-1] = 0
                #         else:
                #             state_matrix[start:end][j-1] -= n_remove
                # else:
                #     state_matrix = state_matrix

                # # for debugging:
                # diff = state_matrix[start:end]-old
                # if any(x != 0 for x in diff):
                    # print("This is what's up with the ADR state matrix bit: " + str(diff))
    # print(state_matrix)

    return state_matrix

def implement_adr_cont(state_matrix, MOCAT, adr_params):
    if len(adr_params.target_species) == 0:
        state_matrix = state_matrix
    elif adr_params.target_species is not None:
        time = adr_params.time
        for i, sp in enumerate(MOCAT.scenario_properties.species_names):
            params = adr_params.properties.get(sp) 
            start = i*MOCAT.scenario_properties.n_shells
            end = (i+1)*MOCAT.scenario_properties.n_shells
            old = state_matrix[start:end]
            max_indices = [0] * params.get("n_max")

            
            # to remove based on percentage:
            if (params.get("target") == 1) and (time >= params.get("t0")):
                # # targeting the shells with the most of the target species:
                # sorted_mat = sorted(state_matrix[start:end], reverse=True)
                # for j in range(len(max_indices)):
                #     jj = j-1
                #     max_indices[jj] = np.where(state_matrix[start:end]==sorted_mat[jj])
                #     state_matrix[start:end][max_indices[jj]] *= (1-params.get("p_remove"))
                for j in params.get("target_shell"):
                    state_matrix[start:end][j-1] *= (1-params.get("p_remove"))
            else:
                state_matrix = state_matrix

            # # to remove based on specific number:
            # if (params.get("target") == 1) and (time >= params.get("t0")):
            #     n_remove = params.get("n_remove")
            #     for j in params.get("target_shell"):
            #         if n_remove > state_matrix[start:end][j-1]:
            #             state_matrix[start:end][j-1] = 0
            #         else:
            #             state_matrix[start:end][j-1] -= n_remove
            # else:
            #     state_matrix = state_matrix

            # # for debugging:
            # diff = state_matrix[start:end]-old
            # if any(x != 0 for x in diff):
                # print("This is what's up with the ADR state matrix bit: " + str(diff))
    # print(state_matrix)

    return state_matrix

def implement_adr2(state_matrix, MOCAT, adr_params):
    if (len(adr_params.target_species) == 0) or (adr_params.target_species is None):
        state_matrix = state_matrix
    elif adr_params.target_species is not None:
        if adr_params.time in adr_params.adr_times:
            for i, sp in enumerate(MOCAT.scenario_properties.species_names):
                if sp in adr_params.target_species:  
                    start = i*MOCAT.scenario_properties.n_shells
                    end = (i+1)*MOCAT.scenario_properties.n_shells
                    old = state_matrix[start:end]
                    # max_indices = [0] * adr_params.n_max
                    # # targeting the shells with the most of the target species:
                    # sorted_mat = sorted(state_matrix[start:end], reverse=True)
                    # for j in range(adr_params.n_max):
                    #     jj = j-1
                    #     max_indices[jj] = np.where(state_matrix[start:end]==sorted_mat[jj])
                        # if "p" in adr_params.remove_method:
                    #     state_matrix[start:end][max_indices[jj]] *= (1-adr_params.p_remove)
                        # elif "n" in adr_params.remove_method:
                        #     if adr_params.n_remove > state_matrix[start:end][max_indices[jj]]:
                        #         state_matrix[start:end][max_indices[jj]] = 0
                        #     else:
                        #         state_matrix[start:end][max_indices[jj]] -= adr_params.n_remove

                    if "p" in adr_params.remove_method:
                        # idx = np.where(adr_params.target_species == sp)
                        idx = adr_params.target_species.index(sp)
                        p_remove = adr_params.p_remove[idx]
                        # p_remove = adr_params.p_remove
                        for j in adr_params.target_shell:
                            state_matrix[start:end][j-1] *= (1-p_remove)
                    elif "n" in adr_params.remove_method:
                        idx = adr_params.target_species.index(sp)
                        n_remove = adr_params.n_remove[idx]
                        for j in adr_params.target_shell:
                            if n_remove > state_matrix[start:end][j-1]:
                                state_matrix[start:end][j-1] = 0
                            else:
                                state_matrix[start:end][j-1] -= n_remove
                else:
                    state_matrix = state_matrix

                # # for debugging:
                # diff = state_matrix[start:end]-old
                # if any(x != 0 for x in diff):
                    # print("This is what's up with the ADR state matrix bit: " + str(diff))
    # print(state_matrix)

    return state_matrix

def optimize_ADR_UMPY(results, UMPY, adr_params):
    test = "test"
#     # load in the multi_sing_species.json file
#     single_json = json.load(open("./OPUS/configuration/multi_single_species.json"))
#     names = ['name1', 'name2', 'name3']
#     adr_values = [20, 40, 50]
#     # then change the input json
#     # then change the simulation name
#     single_json['simulation_name'] = ['name1']
#     for i in names:
#         single_json['opus']['adr'] = adr_values[i]
#         single_json['simulation_name'] = names[i]
#         # run sim
#         iam_solver.iam_solver(names[i], single_json, i, grid_search=False)

    # ADR optimisaition -> # create a grid space for adr values
#     # some thing to solve for the best adr value
#     # run through the adr values, take the output from iam_solver.iam_solver, store it, then use it for the next iteration