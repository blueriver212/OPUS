#!/usr/bin/env python3
"""
Test script to verify controlled re-entry functionality in EconParameters
"""

import numpy as np
import sys
import os

# Add the OPUS directory to the path
sys.path.append('/Users/indigobrownhall/Code/OPUS')

from OPUS.utils.EconParameters import EconParameters

# Earth's radius (m)
R_EARTH = 6.371e6

def test_controlled_reentry():
    """
    Test the controlled re-entry functionality by comparing normal disposal vs controlled re-entry.
    """
    print("="*60)
    print("TESTING CONTROLLED RE-ENTRY FUNCTIONALITY")
    print("="*60)
    
    # Create a simple test configuration
    test_config = {
        "OPUS": {
            "sat_lifetime": 5,
            "disposal_time": 25,
            "discount_rate": 0.05,
            "intercept": 7.5e5,
            "coef": 1.0e2,
            "tax": 0.0,
            "delta_v_cost": 1000,
            "lift_price": 5000,
            "mass": 700,
            "controlled_renetries_only": False  # Test normal disposal first
        }
    }
    
    # Create a mock MOCAT object with some test shells
    class MockMOCAT:
        class ScenarioProperties:
            def __init__(self):
                # Test altitudes: 400km, 600km, 800km, 1000km
                self.n_shells = 4
                self.R0_km = np.array([400, 600, 800, 1000])  # altitudes in km
                self.R0 = R_EARTH + self.R0_km * 1000  # radii in meters
                self.mu = 3.986004418e14  # Earth's gravitational parameter
                self.Dhl = 50e3  # shell height in meters
                self.species_names = ['test_species']
        
        def __init__(self):
            self.scenario_properties = self.ScenarioProperties()
    
    # Test normal disposal
    print("\n1. Testing NORMAL DISPOSAL (controlled_renetries_only = False)")
    print("-" * 50)
    
    econ_params_normal = EconParameters(test_config, MockMOCAT())
    econ_params_normal.calculate_cost_fn_parameters(0.95, "test_scenario")
    
    print("Normal disposal delta-v requirements:")
    for i, alt in enumerate(econ_params_normal.mocat.scenario_properties.R0_km):
        print(f"  {alt}km altitude: {econ_params_normal.total_deorbit_delta_v[i]:.2f} m/s")
    
    # Test controlled re-entry
    print("\n2. Testing CONTROLLED RE-ENTRY (controlled_renetries_only = True)")
    print("-" * 50)
    
    test_config["OPUS"]["controlled_renetries_only"] = True
    econ_params_controlled = EconParameters(test_config, MockMOCAT())
    econ_params_controlled.calculate_cost_fn_parameters(0.95, "test_scenario")
    
    print("Controlled re-entry delta-v requirements (to 75km perigee):")
    for i, alt in enumerate(econ_params_controlled.mocat.scenario_properties.R0_km):
        print(f"  {alt}km altitude: {econ_params_controlled.total_deorbit_delta_v[i]:.2f} m/s")
    
    # Compare results
    print("\n3. COMPARISON")
    print("-" * 50)
    print("Altitude | Normal Disposal | Controlled Re-entry | Difference")
    print("---------|-----------------|---------------------|-----------")
    
    for i, alt in enumerate(econ_params_normal.mocat.scenario_properties.R0_km):
        normal_dv = econ_params_normal.total_deorbit_delta_v[i]
        controlled_dv = econ_params_controlled.total_deorbit_delta_v[i]
        diff = controlled_dv - normal_dv
        print(f"  {alt:3d}km  |     {normal_dv:8.2f}     |      {controlled_dv:8.2f}      |  {diff:+6.2f}")
    
    # Verify that controlled re-entry always requires more delta-v
    print(f"\n4. VERIFICATION")
    print("-" * 50)
    all_controlled_higher = all(controlled >= normal for controlled, normal in 
                               zip(econ_params_controlled.total_deorbit_delta_v, 
                                   econ_params_normal.total_deorbit_delta_v))
    
    if all_controlled_higher:
        print("✓ PASS: Controlled re-entry always requires equal or more delta-v than normal disposal")
    else:
        print("✗ FAIL: Some altitudes show controlled re-entry requiring less delta-v")
    
    # Check that 75km perigee is reasonable
    print(f"\n5. CONTROLLED RE-ENTRY ANALYSIS")
    print("-" * 50)
    print("Controlled re-entry brings perigee to 75km altitude, which should:")
    print("- Ensure atmospheric drag will cause re-entry")
    print("- Require more delta-v than normal disposal (which goes to compliant altitude)")
    print("- Be consistent across all starting altitudes")
    
    return econ_params_normal, econ_params_controlled

if __name__ == "__main__":
    normal_params, controlled_params = test_controlled_reentry()
