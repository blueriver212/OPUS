import numpy as np

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


        if species_name.startswith('Su'):
            # All compliant PMD are dropped at the highest naturall compliant vector. 
            last_compliant_shell = np.where(species.econ_params.naturally_compliant_vector == 1)[0][-1]
            non_compliant_mask = (species.econ_params.naturally_compliant_vector == 0)

            # calculate compliant and non-compliant derelicts - just for reporting
            compliant_derelicts = (1 / species.deltat) * num_items_fringe * species.econ_params.comp_rate
            non_compliant_derelicts = (1 / species.deltat) * num_items_fringe * (1 - species.econ_params.comp_rate)

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

        elif species_name.startswith('S'):
            # 97% removed from sim, 3% fail PMD and get dropped randomly into compliant shells
            successful_pmd = species.econ_params.comp_rate * (1 / species.deltat) * num_items_fringe
            failed_pmd = (1 - species.econ_params.comp_rate) * (1 / species.deltat) * num_items_fringe

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
                successful_pmd = species.econ_params.comp_rate * (1 / species.deltat) * num_items_fringe
                failed_pmd = (1 - species.econ_params.comp_rate) * (1 / species.deltat) * num_items_fringe

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
                compliant_derelicts = (1 / species.deltat) * num_items_fringe * species.econ_params.comp_rate
                non_compliant_derelicts = (1 / species.deltat) * num_items_fringe * (1 - species.econ_params.comp_rate)

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

            # Use comp_rate to determine total compliance (bond-dependent)
            # comp_rate is per-shell array, so we handle it per shell
            comp_rate = species.econ_params.comp_rate
            
            # Calculate total compliant and non-compliant items based on comp_rate
            # comp_rate determines what fraction will comply (controlled + uncontrolled)
            compliant_items_total = comp_rate * items_to_pmd_total  # Per shell array
            non_compliant_items_total = (1 - comp_rate) * items_to_pmd_total  # Per shell array
            
            # Calculate base PMD fractions (these are the baseline proportions)
            total_compliant_pmd = controlled_pmd + uncontrolled_pmd
            total_non_compliant_pmd = no_attempt_pmd + failed_attempt_pmd
            
            # Calculate proportional fractions for distributing compliant items
            if total_compliant_pmd > 0:
                controlled_fraction = controlled_pmd / total_compliant_pmd
                uncontrolled_fraction = uncontrolled_pmd / total_compliant_pmd
            else:
                controlled_fraction = 0
                uncontrolled_fraction = 0
            
            # Calculate proportional fractions for distributing non-compliant items
            if total_non_compliant_pmd > 0:
                no_attempt_fraction = no_attempt_pmd / total_non_compliant_pmd
                failed_attempt_fraction = failed_attempt_pmd / total_non_compliant_pmd
            else:
                no_attempt_fraction = 0
                failed_attempt_fraction = 0

            species.sum_compliant = 0
            species.sum_non_compliant = 0
            
            # Controlled PMD: distribute compliant items proportionally
            if controlled_pmd > 0:
                controlled_derelicts = controlled_fraction * compliant_items_total
                # Remove from active (already done above, but keep for clarity)
                state_matrix[:, species.species_idx, 0] -= controlled_derelicts
                state_matrix_alt[:, species.species_idx] -= controlled_derelicts
                species.sum_compliant += np.sum(controlled_derelicts)

            # Uncontrolled PMD: distribute compliant items proportionally
            if uncontrolled_pmd > 0:
                uncontrolled_items = uncontrolled_fraction * compliant_items_total
                hp = get_disposal_orbits(year, HMid, species_name, pmd_lifetime=species.econ_params.disposal_time)

                # 2) convert to eccentricities
                sma, e = sma_ecc_from_apogee_perigee(hp, HMid)

                # 3) map to your eccentricity bins
                ecc_bin_idx = map_ecc_to_bins(e, eccentricity_bins)
                sma_bin_idx = map_sma_to_bins(sma, sma_bins)

                # Distribute uncontrolled items to disposal orbits
                for i, idx in enumerate(ecc_bin_idx):
                    # if hp is nan, then derelicts just remain in the same shell as a derelict
                    if np.isnan(hp[i]):
                        uncontrolled_amount = uncontrolled_items[i]
                        state_matrix[i, species.derelict_idx, 0] += uncontrolled_amount
                        state_matrix_alt[i, species.derelict_idx] += uncontrolled_amount
                        species.sum_compliant += np.sum(uncontrolled_amount)
                    
                    # if not find the new sma and ecc bin and place the derelicts there
                    else:
                        uncontrolled_amount = uncontrolled_items[i]
                        state_matrix[sma_bin_idx[i], species.derelict_idx, ecc_bin_idx[i]] += uncontrolled_amount
                        state_matrix_alt[sma_bin_idx[i], species.derelict_idx] += uncontrolled_amount
                        species.sum_compliant += np.sum(uncontrolled_amount)

            # No attempt PMD: distribute non-compliant items proportionally
            if no_attempt_pmd > 0:
                no_attempt_amount = no_attempt_fraction * non_compliant_items_total
                # Remove the satellites from the simulation
                state_matrix[:, species.species_idx, 0] -= no_attempt_amount
                state_matrix_alt[:, species.species_idx] -= no_attempt_amount

                # Add the satellites to the derelict slice
                state_matrix[:, species.derelict_idx, 0] += no_attempt_amount
                state_matrix_alt[:, species.derelict_idx] += no_attempt_amount
                
                species.sum_non_compliant += np.sum(no_attempt_amount)
                
            # Failed attempt PMD: distribute non-compliant items proportionally
            if failed_attempt_pmd > 0:
                failed_attempt_amount = failed_attempt_fraction * non_compliant_items_total
                # For now, treat same as no_attempt (remain in same shell as derelict)
                state_matrix[:, species.species_idx, 0] -= failed_attempt_amount
                state_matrix_alt[:, species.species_idx] -= failed_attempt_amount
                state_matrix[:, species.derelict_idx, 0] += failed_attempt_amount
                state_matrix_alt[:, species.derelict_idx] += failed_attempt_amount
                species.sum_non_compliant += np.sum(failed_attempt_amount)

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

def get_disposal_orbits(year, apogees_km, satellite_type, pmd_lifetime=5.0, lookup_path="disposal_lookup_.npz"):
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
   # ------------------------------------------------------------------
    #  Robust Path Resolution: Search Upwards for 'indigo-thesis'
    # ------------------------------------------------------------------
    # Start searching from the directory containing this script
    current_search_dir = os.path.dirname(os.path.abspath(__file__))
    base_lookup_dir = None
    
    # Walk up the directory tree (up to 4 levels) to find the data folder
    for _ in range(4):
        # Check if 'indigo-thesis/disposal-altitude' exists relative to current search dir
        candidate_path = os.path.join(current_search_dir, 'indigo-thesis', 'disposal-altitude')
        
        if os.path.exists(candidate_path):
            base_lookup_dir = candidate_path
            break
        
        # Move up one directory level
        parent_dir = os.path.dirname(current_search_dir)
        if parent_dir == current_search_dir: # We hit the root drive
            break
        current_search_dir = parent_dir
        
    if base_lookup_dir is None:
        # Final fallback: prints a clear error if the folder is missing entirely
        raise FileNotFoundError(
            "Could not locate the 'indigo-thesis/disposal-altitude' directory. "
            "Checked parent directories of the script but could not find the data."
        )

    # ------------------------------------------------------------------
    #  Select specific file based on satellite type
    # ------------------------------------------------------------------
    if satellite_type == "S":
        lookup_path = os.path.join(base_lookup_dir, "disposal_lookup_S.npz")
    elif satellite_type == "Su":
        lookup_path = os.path.join(base_lookup_dir, "disposal_lookup_Su.npz")
    elif satellite_type == "Sns":
        lookup_path = os.path.join(base_lookup_dir, "disposal_lookup_S.npz")
    else:
        raise ValueError(f"Invalid satellite type: {satellite_type}")
    
    L = _load_lookup_cached(lookup_path)

    years = L["years"]
    apogee_grid = L["apogee_alts_km"]
    perigee_grid = L["perigee_alts_km"]
    lifetimes_years = L["lifetimes_years"]  # (ny, na, np)

    # choose nearest available year
    iy = int(np.argmin(np.abs(years - int(year))))

    perigees_out = np.full_like(apogees_km, np.nan, dtype=float)

    for j, ap in enumerate(apogees_km):
        # find closest apogee altitude
        apogee_idx = int(np.argmin(np.abs(apogee_grid - ap)))
        
        # get lifetimes for this apogee
        apogee_lifetimes = lifetimes_years[iy, apogee_idx, :]
        
        # find valid lifetimes
        valid_mask = ~np.isnan(apogee_lifetimes) & (apogee_lifetimes > 0)
        
        if not np.any(valid_mask):
            perigees_out[j] = np.nan
            continue
        
        valid_lifetimes = apogee_lifetimes[valid_mask]
        valid_perigees = perigee_grid[valid_mask]
        
        # interpolate to find perigee for target lifetime
        if pmd_lifetime <= np.min(valid_lifetimes):
            perigee_est = np.min(valid_perigees)
        elif pmd_lifetime >= np.max(valid_lifetimes):
            perigee_est = np.max(valid_perigees)
        else:
            perigee_est = np.interp(pmd_lifetime, valid_lifetimes, valid_perigees)
        
        # Physical constraint: perigee must be <= apogee
        if perigee_est > ap:
            perigees_out[j] = np.nan  # Not possible to dispose to this orbit
        else:
            perigees_out[j] = perigee_est

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