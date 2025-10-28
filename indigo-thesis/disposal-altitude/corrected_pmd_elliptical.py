#!/usr/bin/env python3
"""
Corrected implementation of evaluate_pmd_elliptical for JB2008 density model
"""

import numpy as np
import sys
import os

# Add the OPUS utils to path
sys.path.append('/Users/indigobrownhall/Code/OPUS/OPUS/utils')
from PostMissionDisposal import get_disposal_orbits, sma_ecc_from_apogee_perigee, map_ecc_to_bins, map_sma_to_bins

def corrected_evaluate_pmd_elliptical(state_matrix, state_matrix_alt, multi_species, 
                                    year, density_model, HMid, eccentricity_bins, sma_bins):
    """
    CORRECTED PMD logic for elliptical orbits with JB2008 density model:
    - 'S': Uses controlled_pmd, uncontrolled_pmd, no_attempt_pmd, failed_attempt_pmd
    - 'Su': Uses controlled_pmd, uncontrolled_pmd, no_attempt_pmd, failed_attempt_pmd  
    - 'Sns': no PMD, all become derelicts in place
    """
    
    if density_model == "JB2008_dens_func":
        for species in multi_species.species:
            species_name = species.name

            controlled_pmd = species.econ_params.controlled_pmd
            uncontrolled_pmd = species.econ_params.uncontrolled_pmd
            no_attempt_pmd = species.econ_params.no_attempt_pmd
            failed_attempt_pmd = species.econ_params.failed_attempt_pmd

            # Array of the items to PMD (total satellites at end of life)
            total_species = state_matrix[:, species.species_idx, 0]
            items_to_pmd_total = total_species * (1 / species.deltat)

            # Remove all satellites at end of life from both sma and alt bins
            state_matrix[:, species.species_idx, 0] -= items_to_pmd_total
            state_matrix_alt[:, species.species_idx] -= items_to_pmd_total

            # Initialize counters
            species.sum_compliant = 0
            species.sum_non_compliant = 0
            
            # 1. CONTROLLED PMD: Successfully deorbited satellites (removed from simulation)
            if controlled_pmd > 0:
                controlled_count = items_to_pmd_total * controlled_pmd
                # These are already removed above, just count them
                species.sum_compliant += np.sum(controlled_count)

            # 2. UNCONTROLLED PMD: Satellites that attempt disposal but go to disposal orbits
            if uncontrolled_pmd > 0:
                uncontrolled_count = items_to_pmd_total * uncontrolled_pmd
                
                # Get disposal perigee heights for each altitude bin
                try:
                    hp = get_disposal_orbits(year, HMid, species_name, 
                                           pmd_lifetime=species.econ_params.disposal_time)
                except Exception as e:
                    print(f"Warning: Could not get disposal orbits for {species_name}: {e}")
                    # If disposal lookup fails, treat as no attempt
                    hp = np.full_like(HMid, np.nan)
                
                # Convert perigee heights to orbital elements
                sma, e = sma_ecc_from_apogee_perigee(hp, HMid)
                
                # Map to eccentricity and SMA bins
                ecc_bin_idx = map_ecc_to_bins(e, eccentricity_bins)
                sma_bin_idx = map_sma_to_bins(sma, sma_bins)
                
                # Distribute uncontrolled satellites to disposal orbits
                for i in range(len(HMid)):
                    if not np.isnan(hp[i]) and ecc_bin_idx[i] >= 0 and sma_bin_idx[i] >= 0:
                        # Valid disposal orbit found
                        state_matrix[sma_bin_idx[i], species.derelict_idx, ecc_bin_idx[i]] += uncontrolled_count[i]
                        state_matrix_alt[sma_bin_idx[i], species.derelict_idx] += uncontrolled_count[i]
                        species.sum_compliant += uncontrolled_count[i]
                    else:
                        # No valid disposal orbit, remain as derelict in same shell
                        state_matrix[i, species.derelict_idx, 0] += uncontrolled_count[i]
                        state_matrix_alt[i, species.derelict_idx] += uncontrolled_count[i]
                        species.sum_non_compliant += uncontrolled_count[i]

            # 3. NO ATTEMPT PMD: Satellites that don't attempt disposal (become derelicts in place)
            if no_attempt_pmd > 0:
                no_attempt_count = items_to_pmd_total * no_attempt_pmd
                
                # Add to derelict slice in same altitude bins
                state_matrix[:, species.derelict_idx, 0] += no_attempt_count
                state_matrix_alt[:, species.derelict_idx] += no_attempt_count
                
                species.sum_non_compliant += np.sum(no_attempt_count)
                
            # 4. FAILED ATTEMPT PMD: Satellites that attempt disposal but fail
            if failed_attempt_pmd > 0:
                failed_attempt_count = items_to_pmd_total * failed_attempt_pmd
                
                # These remain as derelicts in the same shell (failed disposal attempt)
                state_matrix[:, species.derelict_idx, 0] += failed_attempt_count
                state_matrix_alt[:, species.derelict_idx] += failed_attempt_count
                
                species.sum_non_compliant += np.sum(failed_attempt_count)

    return state_matrix, state_matrix_alt, multi_species

def test_corrected_implementation():
    """Test the corrected implementation with sample data"""
    
    print("Testing corrected evaluate_pmd_elliptical implementation...")
    
    # Create mock data
    n_alt_bins = 10
    n_species = 3
    n_ecc_bins = 5
    
    state_matrix = np.random.rand(n_alt_bins, n_species, n_ecc_bins) * 100
    state_matrix_alt = np.random.rand(n_alt_bins, n_species) * 100
    
    # Mock species data
    class MockSpecies:
        def __init__(self, name, species_idx, derelict_idx, deltat, disposal_time):
            self.name = name
            self.species_idx = species_idx
            self.derelict_idx = derelict_idx
            self.deltat = deltat
            self.econ_params = MockEconParams(disposal_time)
    
    class MockEconParams:
        def __init__(self, disposal_time):
            self.disposal_time = disposal_time
            self.controlled_pmd = 0.6
            self.uncontrolled_pmd = 0.3
            self.no_attempt_pmd = 0.1
            self.failed_attempt_pmd = 0.0
    
    class MockMultiSpecies:
        def __init__(self):
            self.species = [
                MockSpecies("S", 0, 0, 5, 5),
                MockSpecies("Su", 1, 1, 8, 5),
                MockSpecies("Sns", 2, 2, 10, 5)
            ]
    
    multi_species = MockMultiSpecies()
    year = 2024
    density_model = "JB2008_dens_func"
    HMid = np.linspace(400, 1500, n_alt_bins)
    eccentricity_bins = np.linspace(0, 0.1, n_ecc_bins + 1)
    sma_bins = np.linspace(6778, 7878, n_alt_bins + 1)
    
    print(f"Initial state matrix shape: {state_matrix.shape}")
    print(f"Initial total satellites: {np.sum(state_matrix)}")
    
    # Run corrected function
    try:
        new_state_matrix, new_state_matrix_alt, new_multi_species = corrected_evaluate_pmd_elliptical(
            state_matrix, state_matrix_alt, multi_species, year, density_model, HMid, eccentricity_bins, sma_bins
        )
        
        print(f"Final state matrix shape: {new_state_matrix.shape}")
        print(f"Final total satellites: {np.sum(new_state_matrix)}")
        
        for species in new_multi_species.species:
            print(f"{species.name}: Compliant={species.sum_compliant:.1f}, Non-compliant={species.sum_non_compliant:.1f}")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_corrected_implementation()

