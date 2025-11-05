#!/usr/bin/env python3
"""
Test script for tolerance search - simplified version to test the setup
"""

import sys
import os
import json
import numpy as np
import time
from pathlib import Path

# Add the OPUS directory to the path
sys.path.append('/Users/indigobrownhall/Code/OPUS/OPUS')

from utils.MultiSpecies import MultiSpecies
from utils.MocatParameters import configure_mocat
from utils.MultiSpeciesOpenAccessSolver import MultiSpeciesOpenAccessSolver

def test_single_run():
    """Test a single run with default settings"""
    print("Testing single run with default settings...")
    
    # Configuration
    config_path = "/Users/indigobrownhall/Code/OPUS/OPUS/configuration/multi_single_species.json"
    target_values = np.array([7677, 2665, 1228])  # S, Su, Sns
    species_names = ["S", "Su", "Sns"]
    
    # Load MOCAT configuration
    with open(config_path, 'r') as f:
        MOCAT_config = json.load(f)
    
    try:
        # Initialize multi-species
        multi_species = MultiSpecies(species_names)
        
        # Configure MOCAT
        MOCAT, multi_species = configure_mocat(
            MOCAT_config, 
            multi_species=multi_species, 
            grid_search=True
        )
        
        # Get species position indexes and parameters
        multi_species.get_species_position_indexes(MOCAT)
        multi_species.get_mocat_species_parameters(MOCAT)
        
        print("✓ MOCAT configuration loaded successfully")
        print(f"✓ Species names: {MOCAT.scenario_properties.species_names}")
        print(f"✓ Elliptical: {MOCAT.scenario_properties.elliptical}")
        print(f"✓ Number of shells: {MOCAT.scenario_properties.n_shells}")
        
        # Get initial state
        x0 = MOCAT.scenario_properties.x0
        elliptical = MOCAT.scenario_properties.elliptical
        
        # Flatten for circular orbits
        if not elliptical:
            x0 = x0.T.values.flatten()
        
        print(f"✓ Initial state shape: {x0.shape}")
        
        # Create solver guess
        solver_guess = x0.copy()
        lam = np.full_like(x0, None, dtype=object)
        
        if elliptical:
            for species in multi_species.species:
                initial_guess = 0.05 * x0[:, species.species_idx, 0]
                initial_guess = np.maximum(initial_guess, 0.0)
                if np.sum(initial_guess) == 0:
                    initial_guess[:] = 5
                lam[:, species.species_idx, 0] = initial_guess
                solver_guess[:, species.species_idx, 0] = initial_guess
        else:
            for species in multi_species.species:
                initial_guess = 0.05 * np.array(x0[species.start_slice:species.end_slice])
                if np.sum(initial_guess) == 0:
                    initial_guess[:] = 5
                solver_guess[species.start_slice:species.end_slice] = initial_guess
                lam[species.start_slice:species.end_slice] = initial_guess
        
        print("✓ Solver guess created")
        
        # Test with loose tolerances
        years = [2017 + i for i in range(MOCAT.scenario_properties.simulation_duration)]
        
        custom_solver_options = {
            'method': 'trf',
            'verbose': 0,
            'ftol': 1e-3,
            'xtol': 0.1,
            'gtol': 1e-3,
            'max_nfev': 100
        }
        
        print("✓ Custom solver options created")
        
        open_access = MultiSpeciesOpenAccessSolver(
            MOCAT, solver_guess, x0, "linear", lam, multi_species, years, 0, custom_solver_options
        )
        
        print("✓ MultiSpeciesOpenAccessSolver created")
        
        # Run solver
        start_time = time.time()
        launch_rate, col_probability_all_species, umpy, excess_returns, last_non_compliance = open_access.solver()
        computation_time = time.time() - start_time
        
        print(f"✓ Solver completed in {computation_time:.2f} seconds")
        print(f"✓ Launch rate shape: {launch_rate.shape}")
        print(f"✓ Launch rate sum: {np.sum(launch_rate)}")
        
        # Calculate final species counts
        final_counts = {}
        for i, species_name in enumerate(species_names):
            if elliptical:
                species_data = x0[:, i, :]
                final_counts[species_name] = np.sum(species_data)
            else:
                species_slice = slice(i * MOCAT.scenario_properties.n_shells, 
                                    (i + 1) * MOCAT.scenario_properties.n_shells)
                final_counts[species_name] = np.sum(x0[species_slice])
        
        print(f"✓ Final counts: {final_counts}")
        
        # Calculate accuracy
        predicted_values = np.array([final_counts[name] for name in species_names])
        target_safe = np.where(target_values == 0, 1e-10, target_values)
        mape = np.mean(np.abs((predicted_values - target_values) / target_safe)) * 100
        
        print(f"✓ Accuracy (MAPE): {mape:.2f}%")
        print(f"✓ Target values: {target_values}")
        print(f"✓ Predicted values: {predicted_values}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Tolerance Search Setup")
    print("="*50)
    
    success = test_single_run()
    
    if success:
        print("\n✓ Test completed successfully!")
        print("You can now run the full tolerance search with:")
        print("python tolerance_search.py")
    else:
        print("\n✗ Test failed. Please check the error messages above.")
