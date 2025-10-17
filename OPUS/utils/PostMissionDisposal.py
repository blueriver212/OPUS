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
 

def evaluate_pmd_elliptical(state_matrix, state_matrix_alt, multi_species, 
                            year, density_model, HMid, eccentricity_bins, sma_bins):
    """
    PMD logic applied per species type:
    - 'S': 97% removed, 3% go to top compliant shell
    - 'Su': current logic (PMD-compliant to last compliant shell, non-compliant stay in shell)
    - 'Sns': no PMD, all become derelicts in place
    """
    
    if density_model == "static_exp_dens_func":
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
    
    if density_model == "JB2008_dens_func":
        for species in multi_species.species:
            species_name = species.name

            controlled_pmd = species.econ_params.controlled_pmd
            uncontrolled_pmd = species.econ_params.uncontrolled_pmd
            no_attempt_pmd = species.econ_params.no_attempt_pmd
            failed_attempt_pmd = species.econ_params.failed_attempt_pmd

            # array of the items to pmd            
            total_species = state_matrix[:, species.species_idx, 0]

            items_to_pmd_total = total_species * (1 / species.deltat)

            # Remove all satellites at end of life - from both sma and alt bins
            state_matrix[:, species.species_idx, 0] -= items_to_pmd_total
            state_matrix_alt[:, species.species_idx] -= items_to_pmd_total
            
            # controlled satellites just get removed from the simulation
            if controlled_pmd > 0:
                # of the derelicts remove those that are controlled 
                # items_to_pmd_total = items_to_pmd_total * (1 - controlled_pmd)
                continue

            if uncontrolled_pmd > 0:
                hp = get_disposal_orbits(year, HMid, pmd_lifetime=species.econ_params.disposal_time,
                         lookup_path="disposal_lookup_24_26.npz")  # or .mat

                # 2) convert to eccentricities
                sma, e = sma_ecc_from_apogee_perigee(hp, HMid)

                # 3) map to your eccentricity bins
                ecc_bin_idx = map_ecc_to_bins(e, eccentricity_bins)
                sma_bin_idx = map_sma_to_bins(sma, sma_bins)

                # this will give you a map of where to go for the number of uncontrolled reentries
                for derelicts, idx in enumerate(ecc_bin_idx):
                    # if hp is nan, then derelicts just remain in the same shell as a derelict
                    if hp[idx] is np.nan:
                        state_matrix[idx, species.derelict_idx, 0] += items_to_pmd_total * (1 - uncontrolled_pmd)
                        state_matrix_alt[idx, species.derelict_idx] += items_to_pmd_total * (1 - uncontrolled_pmd)
                    
                    # if not find the new sma and ecc bin and place the derelicts there
                    else:
                        state_matrix[sma_bin_idx[idx], species.derelict_idx, ecc_bin_idx[idx]] += items_to_pmd_total[idx] * (1 - uncontrolled_pmd)
                        state_matrix_alt[sma_bin_idx[idx], species.derelict_idx] += items_to_pmd_total[idx] * (1 - uncontrolled_pmd)

            # if no attempt derelicts, then they remain in the same shell as a derelict
            if no_attempt_pmd > 0:
                state_matrix[:, species.derelict_idx, 0] += items_to_pmd_total * no_attempt_pmd
                state_matrix_alt[:, species.derelict_idx] += items_to_pmd_total * no_attempt_pmd

            if failed_attempt_pmd > 0:
                continue # to implement

    return state_matrix, state_matrix_alt, multi_species


import os
import numpy as np
import scipy.io as sio

RE_KM = 6378.136  # Earth radius [km]

# ---------- internal helpers ----------

def _load_lookup_cached(lookup_path: str):
    """
    Loads disposal_lookup either from .npz (fast) or .mat (then caches to .npz).
    Returns a plain dict of numpy arrays.
    """
    base, ext = os.path.splitext(lookup_path)
    npz_path = base + ".npz"

    # If .npz exists & is newer than source, use it
    if os.path.exists(npz_path) and (not os.path.exists(lookup_path) or
                                     os.path.getmtime(npz_path) >= os.path.getmtime(lookup_path)):
        z = np.load(npz_path)
        return {k: z[k] for k in z.files}

    # Otherwise load .mat and cache
    if ext.lower() == ".mat" or not os.path.exists(npz_path):
        data = sio.loadmat(lookup_path, squeeze_me=True, struct_as_record=False)
        L = data["lookup"]
        out = {
            "years":            np.array(L.years, dtype=int).flatten(),
            "apogee_alts_km":   np.array(L.apogee_alts_km).flatten(),
            "perigee_alts_km":  np.array(L.perigee_alts_km).flatten(),
            "coef_logquad":     np.array(L.coef_logquad),  # (ny, na, 3)
            "R2_log":           np.array(L.R2_log),
            "lifetimes_years":  np.array(L.lifetimes_years),
            "decay_alt_km":     np.array(L.decay_alt_km).item() if np.size(L.decay_alt_km)==1 else np.array(L.decay_alt_km),
        }
        # cache
        np.savez(npz_path, **out)
        return out

    # Fall-through (shouldn't happen): try npz
    z = np.load(npz_path)
    return {k: z[k] for k in z.files}


def _inv_logquad_for_y(p, y_target, xmin, xmax):
    """
    Given p=[a,b,c] with log(y)=a x^2 + b x + c, return x (perigee km)
    such that y = y_target. Chooses in-range root closest to middle.
    """
    if p is None or np.any(np.isnan(p)):
        return np.nan
    a, b, c = p
    c = c - np.log(y_target)
    D = b**2 - 4*a*c
    if D < 0:
        return np.nan
    roots = np.array([(-b + np.sqrt(D)) / (2*a), (-b - np.sqrt(D)) / (2*a)], dtype=float)
    roots = roots[(roots >= xmin) & (roots <= xmax)]
    if roots.size == 0:
        return np.nan
    return roots[np.argmin(np.abs(roots - 0.5 * (xmin + xmax)))]


# ---------- 1) main function you asked for ----------

def get_disposal_orbits(year, apogees_km, pmd_lifetime=5.0, lookup_path="disposal_lookup_24_26.npz"):
    """
    Parameters
    ----------
    year : int
        Start year to use (e.g. 2024, 2026, 2028...).
    apogees_km : array-like
        Array of apogee altitudes (km) for which you want perigee targets.
    pmd_lifetime : float, optional
        Desired disposal lifetime in years (default 5.0).
    lookup_path : str
        Path to lookup ('.npz' or '.mat' – if '.mat', it will cache to '.npz').

    Returns
    -------
    perigees_km : np.ndarray
        Perigee altitudes (km) matching apogees_km. NaN where no valid solution.
    """
    apogees_km = np.asarray(apogees_km, dtype=float).ravel()
    L = _load_lookup_cached(lookup_path)

    years         = L["years"]
    apogee_grid   = L["apogee_alts_km"]
    perigee_grid  = L["perigee_alts_km"]
    coef_logquad  = L["coef_logquad"]      # (ny, na, 3)
    R2_log        = L["R2_log"]

    # choose nearest available year
    iy = int(np.argmin(np.abs(years - int(year))))

    hp_min, hp_max = float(perigee_grid.min()), float(perigee_grid.max())
    perigees_out = np.full_like(apogees_km, np.nan, dtype=float)

    for j, ap in enumerate(apogees_km):
        # find bracket on apogee grid
        i1 = np.searchsorted(apogee_grid, ap, side="left")
        if i1 == 0:
            i0 = i1
        elif i1 >= len(apogee_grid):
            i0 = len(apogee_grid) - 1
            i1 = i0
        else:
            i0 = i1 - 1

        ap0 = apogee_grid[i0]
        p0  = coef_logquad[iy, i0, :]
        hp0 = _inv_logquad_for_y(p0, pmd_lifetime, hp_min, hp_max)

        if i1 == i0:
            hp_est = hp0
        else:
            ap1 = apogee_grid[i1]
            p1  = coef_logquad[iy, i1, :]
            hp1 = _inv_logquad_for_y(p1, pmd_lifetime, hp_min, hp_max)
            # linear interp across apogee dimension (only if both valid)
            if np.isnan(hp0) and np.isnan(hp1):
                hp_est = np.nan
            elif np.isnan(hp0):
                hp_est = hp1
            elif np.isnan(hp1):
                hp_est = hp0
            else:
                t = (ap - ap0) / (ap1 - ap0) if ap1 != ap0 else 0.0
                hp_est = (1 - t) * hp0 + t * hp1

        # physical checks: perigee must be <= apogee & within fit bounds
        if np.isnan(hp_est) or (hp_est > ap) or (hp_est < hp_min) or (hp_est > hp_max):
            perigees_out[j] = np.nan
        else:
            # Optional: reject very low-quality fits
            if R2_log.size:
                r2_ok = False
                # check neighboring apogee R^2s
                for idx in {i0, i1}:
                    r2 = R2_log[iy, idx]
                    if not np.isnan(r2) and r2 >= 0.90:  # relax/raise as you like
                        r2_ok = True
                        break
                if not r2_ok:
                    perigees_out[j] = np.nan
                    continue
            perigees_out[j] = hp_est

    return perigees_out


# ---------- 2) eccentricity utilities ----------

def sma_ecc_from_apogee_perigee(perigees_km, apogees_km, re_km: float = RE_KM):
    """
    Vectorized: perigee/apogee *altitudes* -> eccentricity e.
    e = (ra - rp) / (ra + rp), with rp = re + hp, ra = re + ha.
    """
    perigees_km = np.asarray(perigees_km, dtype=float)
    apogees_km  = np.asarray(apogees_km, dtype=float)
    rp = re_km + perigees_km
    ra = re_km + apogees_km
    with np.errstate(invalid='ignore', divide='ignore'):
        e = (ra - rp) / (ra + rp)
    # clamp tiny negatives from numerical noise
    e = np.where(e < 0, 0.0, e)

    sma = (ra + rp) / 2
    return sma, e


def map_ecc_to_bins(e_values, ecc_edges):
    """
    Map each eccentricity in e_values to a bin index given edges (like numpy.digitize).
    Returns 0..len(ecc_edges)-2 for in-range, and -1 for NaN/out-of-range.
    """
    e_values = np.asarray(e_values, dtype=float)
    edges = np.asarray(ecc_edges, dtype=float)
    idx = np.digitize(e_values, edges, right=False) - 1
    # mark out-of-range as -1
    bad = (idx < 0) | (idx >= len(edges)-1) | ~np.isfinite(e_values)
    idx[bad] = -1
    return idx

def map_sma_to_bins(sma_values, sma_edges):
    """
    Map each semi-major axis in sma_values to a bin index given edges (like numpy.digitize).
    Returns 0..len(sma_edges)-2 for in-range, and -1 for NaN/out-of-range.
    """
    sma_values = np.asarray(sma_values, dtype=float)
    edges = np.asarray(sma_edges, dtype=float)
    idx = np.digitize(sma_values, edges, right=False) - 1
    # mark out-of-range as -1
    bad = (idx < 0) | (idx >= len(edges)-1) | ~np.isfinite(sma_values)
    idx[bad] = -1
    return idx