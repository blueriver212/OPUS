#!/usr/bin/env python3
"""
Test script for tolerance search v2 - uses existing iam_solver method
"""

import sys
import os
import json
import numpy as np
import time
from pathlib import Path

# Add the OPUS directory to the path
sys.path.append('/Users/indigobrownhall/Code/OPUS/OPUS')

from main import IAMSolver

def test_single_run():
    """Test a single run with default settings using existing iam_solver"""
    print("Testing single run with existing iam_solver method...")
    
    # Configuration
    config_path = "/Users/indigobrownhall/Code/OPUS/OPUS/configuration/multi_single_species.json"
    target_values = np.array([7677, 2665, 1228])  # S, Su, Sns
    
    # Load MOCAT configuration
    with open(config_path, 'r') as f:
        MOCAT_config = json.load(f)
    
    try:
        # Create IAMSolver instance
        iam_solver = IAMSolver()
        
        print("✓ IAMSolver created successfully")
        
        # Run the simulation using the existing iam_solver
        start_time = time.time()
        species_data = iam_solver.iam_solver("Baseline", MOCAT_config, "tolerance_test", grid_search=True)
        computation_time = time.time() - start_time
        
        print(f"✓ Simulation completed in {computation_time:.2f} seconds")
        
        # Calculate final species counts
        final_counts = {}
        species_names = ["S", "Su", "Sns"]
        
        for species_name in species_names:
            if species_name in species_data and isinstance(species_data[species_name], np.ndarray):
                final_counts[species_name] = np.sum(species_data[species_name][-1])
                print(f"✓ {species_name}: {final_counts[species_name]:.0f}")
            else:
                final_counts[species_name] = 0.0
                print(f"✗ {species_name}: No data found")
        
        # Calculate accuracy
        predicted_values = np.array([final_counts[name] for name in species_names])
        target_safe = np.where(target_values == 0, 1e-10, target_values)
        mape = np.mean(np.abs((predicted_values - target_values) / target_safe)) * 100
        
        print(f"✓ Final counts: {final_counts}")
        print(f"✓ Target values: {target_values}")
        print(f"✓ Predicted values: {predicted_values}")
        print(f"✓ Accuracy (MAPE): {mape:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Tolerance Search Setup (Version 2)")
    print("="*50)
    
    success = test_single_run()
    
    if success:
        print("\n✓ Test completed successfully!")
        print("You can now run the full tolerance search with:")
        print("python tolerance_search_v2.py")
    else:
        print("\n✗ Test failed. Please check the error messages above.")

