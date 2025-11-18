import numpy as np
import os
# loosely based on PMD function

def implement_adr(state_matrix, MOCAT, adr_params):
    num_removed = {}
    if (len(adr_params.target_species) == 0) or (adr_params.target_species is None):
        state_matrix = state_matrix
    elif adr_params.target_species is not None:
        if adr_params.time in adr_params.adr_times:
            for i, sp in enumerate(MOCAT.scenario_properties.species_names):
                if sp in adr_params.target_species:  
                    start = i*MOCAT.scenario_properties.n_shells
                    end = (i+1)*MOCAT.scenario_properties.n_shells
                    old = state_matrix[start:end]
                    num = []
                    # max_indices = [0] * adr_params.n_max
                    # # targeting the shells with the most of the target species:
                    # add a statement that says what shell is being targeted !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # add option for targeting shell with highest prob. collision !!!!!!!!!!!
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
                            num.append(state_matrix[start:end][j-1]*p_remove)
                            state_matrix[start:end][j-1] *= (1-p_remove)
                        
                    elif "n" in adr_params.remove_method:
                        idx = adr_params.target_species.index(sp)
                        n_remove = adr_params.n_remove[idx]
                        for j in adr_params.target_shell:
                            if adr_params.removals_left < n_remove:
                                n_remove = adr_params.removals_left
                            if n_remove > state_matrix[start:end][j-1]:
                                n = state_matrix[start:end][j-1] - n_remove
                                n_remove += n
                                state_matrix[start:end][j-1] = 0
                            else:
                                state_matrix[start:end][j-1] -= n_remove
                            num.append(n_remove)
                    num_removed[sp] = {"num_removed":int(np.sum(num))}
                else:
                    state_matrix = state_matrix

                # # for debugging:
                # diff = state_matrix[start:end]-old
                # if any(x != 0 for x in diff):
                    # print("This is what's up with the ADR state matrix bit: " + str(diff))
    # print(state_matrix)

    return state_matrix, num_removed

def optimize_ADR_removal(state_matrix, MOCAT, adr_params):
    test = "test"
    num_removed = {}
    removal_dict = {}
    indicator = 0
    if (len(adr_params.target_species) == 0) or (adr_params.target_species is None):
        state_matrix = state_matrix
    elif adr_params.target_species is not None:
        if adr_params.time in adr_params.adr_times:
            for i, sp in enumerate(MOCAT.scenario_properties.species_names):
                if sp in adr_params.target_species:  
                    n_shells = MOCAT.scenario_properties.n_shells
                    start = i*n_shells
                    end = (i+1)*n_shells
                    old = state_matrix[start:end]
                    num = []
                    # max_indices = [0] * adr_params.n_max
                    # # targeting the shells with the most of the target species:
                    # add a statement that says what shell is being targeted !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # add option for targeting shell with highest prob. collision !!!!!!!!!!!
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
                            # num.append(state_matrix[start:end][j-1]*p_remove)
                            state_matrix[start:end][j-1] *= (1-p_remove)
                        
                    elif "n" in adr_params.remove_method:
                        counter = 1
                        indicator = 0
                        for shell in adr_params.target_shell:
                            idx = adr_params.target_species.index(sp)
                            n_remove = adr_params.removals_left
                            ts = shell
                            if n_remove <= 0:
                                print('No ADR due to budget constraints.')
                                removal_dict[str(ts)] = {}
                                removal_dict[str(ts)]['Implemented'] = 0
                                indicator = 1
                            elif n_remove > state_matrix[start:end][ts- 1]:
                                n = state_matrix[start:end][ts-1] - n_remove
                                removal_dict[str(ts)] = {}
                                removal_dict[str(ts)]['Implemented'] = 1
                                removal_dict[str(ts)]['Exhausted'] = 0
                                removal_dict[str(ts)]['amount_removed'] = int(n_remove + n)
                                removal_dict[str(ts)]['Order'] = int(counter)
                                removal_dict[str(ts)]['Removals_Left'] = int(n*(-1))
                                n_remove = n * (-1)
                                state_matrix[start:end][ts-1] = 0
                                while indicator < 1:
                                    for ii in adr_params.shell_order:
                                        if ((ii not in adr_params.target_shell) and (ii <= n_shells)):
                                            if n_remove > state_matrix[start:end][ii-1]:
                                                counter = counter + 1
                                                n = state_matrix[start:end][ii-1] - n_remove
                                                if n > 0:
                                                    print('I know whats going on here. I know whats going on here. Okay? I do. And if you want me to wander backstage to spill the beans... Im the only one out of the loop, it would seem... and if we check my point total here— I dont NEED to walk to the front, because I know what it is. Its a big ol GOOSE EGG, GANG. Its a FAT ZERO. HELLO!! A little LATE ADDITION to the numerical symbol chart brought to us from our friends in Arabia, a little bit of trivia that I happen to know about the history of numbers. That kind of little tidbit would serve me well in most trivia games, unless it had been RIGGED FROM THE BEGINNING! Oh, Ive only just BEGUN to pull the thread on this sweater, friends. You would THINK in a game where there are only TWO possible correct choices, that one would STUMBLE INTO the right answer every so often, wouldnt you? In fact, the probability of NEVER guessing right in the full game is a STATISTICAL WONDER, and yet, HERE WE ARE. Introduced at the top of the game as a champion, what do you think that means? Icarus, flying too close to the sun. But it seems Daedalus, our little master crafter over here, had some wax wings of his own, didnt he? Wanted to see his son fall. Fall from the sky. Oh, how CLOSE TO THE SUN he flew! Well Im NOT HAVING IT. I solved your labyrinth, puzzle master! The minotaurs escaped and youre gonna get the horns, buddy! I CANNOT WIN!')
                                                    removal_dict[str(ii)] = {}
                                                    removal_dict[str(ii)]['Implemented'] = 0
                                                    removal_dict[str(ii)]['Exhausted'] = 0
                                                    removal_dict[str(ii)]['amount_removed'] = int(n_remove)
                                                    removal_dict[str(ii)]['n'] = int(n)
                                                    removal_dict[str(ii)]['counter'] = int(counter)
                                                    removal_dict[str(ii)]['status'] = 'found a problem'
                                                    indicator = 1
                                                
                                                removal_dict[str(ii)] = {}
                                                removal_dict[str(ii)]['Implemented'] = 1
                                                removal_dict[str(ii)]['Exhausted'] = 0
                                                removal_dict[str(ii)]['amount_removed'] = int(n_remove + n)
                                                removal_dict[str(ii)]['Order'] = int(counter)
                                                removal_dict[str(ii)]['Removals_Left'] = int(n*(-1))
                                                if n_remove == 0 or n == 0:
                                                    indicator = 1
                                                n_remove = n * (-1)
                                                
                                                state_matrix[start:end][ii-1] = 0
                                            else:
                                                counter = counter + 1
                                                state_matrix[start:end][ii-1] -= n_remove
                                                removal_dict[str(ii)] = {}
                                                removal_dict[str(ii)]['Implemented'] = 1
                                                removal_dict[str(ii)]['Exhausted'] = 0 
                                                removal_dict[str(ii)]['amount_removed'] = int(n_remove)
                                                removal_dict[str(ii)]['Order'] = int(counter)
                                                removal_dict[str(ii)]['Removals_Left'] = int(0)
                                                n_remove = 0
                                                indicator = 1
                                        if counter > n_shells:
                                            removal_dict['final'] = {}
                                            removal_dict['final']['Exhausted'] = 1 
                                            removal_dict['final']['Removals Left'] = int(n_remove)
                                            removal_dict['final']['Counter_Status'] = int(counter)
                                            indicator = 1
                            else:
                                state_matrix[start:end][ts-1] -= n_remove
                                shell_num = str(ts)
                                removal_dict[shell_num] = {}
                                removal_dict[shell_num]['Shell'] = shell_num,
                                removal_dict[shell_num]['amount_removed'] = int(n_remove)
                                removal_dict[shell_num]['Order'] = int(counter) 
                                removal_dict[shell_num]['Removals_Left'] = int(0)

                            
                    
                    # num_removed[sp] = {"num_removed":int(np.sum(num))}
                else:
                    state_matrix = state_matrix
                    removal_dict[str(0)] = {}
                    removal_dict[str(0)]['Implemented'] = 0

                # # for debugging:
                # diff = state_matrix[start:end]-old
                # if any(x != 0 for x in diff):
                    # print("This is what's up with the ADR state matrix bit: " + str(diff))
    # print(state_matrix)

    return state_matrix, removal_dict
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

