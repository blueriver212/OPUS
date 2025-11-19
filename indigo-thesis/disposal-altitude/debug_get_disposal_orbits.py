#!/usr/bin/env python3
"""
Debug script to investigate why get_disposal_orbits returns NaNs
"""

import numpy as np
import scipy.io as sio
import sys
import os

# Add the OPUS utils to path
sys.path.append('/Users/indigobrownhall/Code/OPUS/OPUS/utils')
from PostMissionDisposal import get_disposal_orbits, _load_lookup_cached

def debug_mat_file_structure():
    """Debug the structure of the .mat files"""
    
    print("=== Debugging .mat file structure ===")
    
    # Load S data
    print("\n--- S Satellite Data ---")
    s_data = sio.loadmat('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_S.mat', squeeze_me=True, struct_as_record=False)
    lookup_s = s_data['lookup']
    
    print(f"Available fields: {list(lookup_s.dtype.names) if hasattr(lookup_s, 'dtype') else 'Not a structured array'}")
    
    # Check each field
    for field in ['years', 'apogee_alts_km', 'perigee_alts_km', 'lifetimes_years', 'coef_logquad', 'R2_log', 'circ_lifetime_years', 'decay_alt_km']:
        if hasattr(lookup_s, field):
            value = getattr(lookup_s, field)
            print(f"{field}: shape={np.array(value).shape}, dtype={np.array(value).dtype}")
            if field in ['coef_logquad', 'R2_log']:
                print(f"  Sample values: {np.array(value)[:2] if np.array(value).size > 0 else 'Empty'}")
        else:
            print(f"{field}: NOT FOUND")
    
    # Load Su data
    print("\n--- Su Satellite Data ---")
    su_data = sio.loadmat('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_Su.mat', squeeze_me=True, struct_as_record=False)
    lookup_su = su_data['lookup']
    
    print(f"Available fields: {list(lookup_su.dtype.names) if hasattr(lookup_su, 'dtype') else 'Not a structured array'}")
    
    # Check each field
    for field in ['years', 'apogee_alts_km', 'perigee_alts_km', 'lifetimes_years', 'coef_logquad', 'R2_log', 'circ_lifetime_years', 'decay_alt_km']:
        if hasattr(lookup_su, field):
            value = getattr(lookup_su, field)
            print(f"{field}: shape={np.array(value).shape}, dtype={np.array(value).dtype}")
            if field in ['coef_logquad', 'R2_log']:
                print(f"  Sample values: {np.array(value)[:2] if np.array(value).size > 0 else 'Empty'}")
        else:
            print(f"{field}: NOT FOUND")

def debug_load_lookup_cached():
    """Debug the _load_lookup_cached function"""
    
    print("\n=== Debugging _load_lookup_cached function ===")
    
    # Test with S data
    print("\n--- Testing S data ---")
    try:
        s_lookup = _load_lookup_cached('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_S.mat')
        print(f"Keys in S lookup: {list(s_lookup.keys())}")
        for key, value in s_lookup.items():
            print(f"{key}: shape={np.array(value).shape}, dtype={np.array(value).dtype}")
            if key in ['coef_logquad', 'R2_log']:
                print(f"  Sample values: {np.array(value)[:2] if np.array(value).size > 0 else 'Empty'}")
    except Exception as e:
        print(f"Error loading S data: {e}")
    
    # Test with Su data
    print("\n--- Testing Su data ---")
    try:
        su_lookup = _load_lookup_cached('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_Su.mat')
        print(f"Keys in Su lookup: {list(su_lookup.keys())}")
        for key, value in su_lookup.items():
            print(f"{key}: shape={np.array(value).shape}, dtype={np.array(value).dtype}")
            if key in ['coef_logquad', 'R2_log']:
                print(f"  Sample values: {np.array(value)[:2] if np.array(value).size > 0 else 'Empty'}")
    except Exception as e:
        print(f"Error loading Su data: {e}")

def debug_get_disposal_orbits_step_by_step():
    """Debug get_disposal_orbits step by step"""
    
    print("\n=== Debugging get_disposal_orbits step by step ===")
    
    # Test with S data
    print("\n--- Testing S satellites ---")
    try:
        # Test the function with a simple case
        result = get_disposal_orbits(2024, 1000, "S", pmd_lifetime=5.0)
        print(f"Result for S: {result}")
    except Exception as e:
        print(f"Error with S: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with Su data
    print("\n--- Testing Su satellites ---")
    try:
        result = get_disposal_orbits(2024, 1000, "Su", pmd_lifetime=5.0)
        print(f"Result for Su: {result}")
    except Exception as e:
        print(f"Error with Su: {e}")
        import traceback
        traceback.print_exc()

def create_fixed_get_disposal_orbits():
    """Create a fixed version that works with the actual data structure"""
    
    print("\n=== Creating fixed version ===")
    
    def fixed_get_disposal_orbits(year, apogees_km, satellite_type, pmd_lifetime=5.0):
        """Fixed version that works with the actual .mat file structure"""
        
        # Load the appropriate .mat file
        if satellite_type == "S":
            mat_path = '/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_S.mat'
        elif satellite_type == "Su":
            mat_path = '/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_Su.mat'
        else:
            raise ValueError(f"Invalid satellite type: {satellite_type}")
        
        # Load data directly
        data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        lookup = data['lookup']
        
        years = np.array(lookup.years, dtype=int).flatten()
        apogee_alts = np.array(lookup.apogee_alts_km).flatten()
        perigee_alts = np.array(lookup.perigee_alts_km).flatten()
        lifetimes = np.array(lookup.lifetimes_years)  # (ny, na, np)
        
        # Find year index
        year_idx = int(np.argmin(np.abs(years - int(year))))
        
        apogees_km = np.asarray(apogees_km, dtype=float).ravel()
        perigees_out = np.full_like(apogees_km, np.nan, dtype=float)
        
        for j, apogee in enumerate(apogees_km):
            # Find closest apogee altitude
            apogee_idx = int(np.argmin(np.abs(apogee_alts - apogee)))
            
            # Get lifetimes for this apogee
            apogee_lifetimes = lifetimes[year_idx, apogee_idx, :]
            
            # Find valid lifetimes
            valid_mask = ~np.isnan(apogee_lifetimes) & (apogee_lifetimes > 0)
            
            if not np.any(valid_mask):
                perigees_out[j] = np.nan
                continue
            
            valid_lifetimes = apogee_lifetimes[valid_mask]
            valid_perigees = perigee_alts[valid_mask]
            
            # Interpolate to find perigee for target lifetime
            if pmd_lifetime <= np.min(valid_lifetimes):
                perigees_out[j] = np.min(valid_perigees)
            elif pmd_lifetime >= np.max(valid_lifetimes):
                perigees_out[j] = np.max(valid_perigees)
            else:
                perigees_out[j] = np.interp(pmd_lifetime, valid_lifetimes, valid_perigees)
        
        return perigees_out
    
    # Test the fixed version
    print("Testing fixed version...")
    
    # Test S satellites
    result_s = fixed_get_disposal_orbits(2024, [500, 1000, 1500], "S", pmd_lifetime=5.0)
    print(f"S satellites result: {result_s}")
    
    # Test Su satellites
    result_su = fixed_get_disposal_orbits(2024, [500, 1000, 1500], "Su", pmd_lifetime=5.0)
    print(f"Su satellites result: {result_su}")
    
    return fixed_get_disposal_orbits

if __name__ == "__main__":
    debug_mat_file_structure()
    debug_load_lookup_cached()
    debug_get_disposal_orbits_step_by_step()
    fixed_func = create_fixed_get_disposal_orbits()


