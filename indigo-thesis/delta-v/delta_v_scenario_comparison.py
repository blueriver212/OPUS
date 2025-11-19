#!/usr/bin/env python3
"""
Delta-v Scenario Comparison: Normal Disposal vs Controlled Re-entry

This script uses the actual EconParameters code to compare delta-v requirements
between normal disposal and controlled re-entry scenarios across different altitudes.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the OPUS directory to the path
sys.path.append('/Users/indigobrownhall/Code/OPUS')

from OPUS.utils.EconParameters import EconParameters

# Earth's radius (m)
R_EARTH = 6.371e6

def create_mock_mocat(altitudes_km):
    """
    Create a mock MOCAT object with specified altitudes.
    """
    class MockMOCAT:
        class ScenarioProperties:
            def __init__(self, altitudes_km):
                self.n_shells = len(altitudes_km)
                self.R0_km = np.array(altitudes_km)  # altitudes in km
                self.R0 = R_EARTH + self.R0_km * 1000  # radii in meters
                self.mu = 3.986004418e14  # Earth's gravitational parameter
                self.Dhl = 50e3  # shell height in meters
                self.species_names = ['test_species']
        
        def __init__(self, altitudes_km):
            self.scenario_properties = self.ScenarioProperties(altitudes_km)
    
    return MockMOCAT(altitudes_km)

def compare_scenarios(altitudes_km, disposal_time=25):
    """
    Compare normal disposal vs controlled re-entry scenarios.
    """
    print("="*80)
    print("DELTA-V SCENARIO COMPARISON: NORMAL DISPOSAL vs CONTROLLED RE-ENTRY")
    print("="*80)
    print(f"Analyzing altitudes: {altitudes_km} km")
    print(f"Disposal time requirement: {disposal_time} years")
    
    # Create mock MOCAT
    mocat = create_mock_mocat(altitudes_km)
    
    # Base configuration
    base_config = {
        "OPUS": {
            "sat_lifetime": 5,
            "disposal_time": disposal_time,
            "discount_rate": 0.05,
            "intercept": 7.5e5,
            "coef": 1.0e2,
            "tax": 0.0,
            "delta_v_cost": 1000,
            "lift_price": 5000,
            "mass": 700,
            "controlled_renetries_only": False
        }
    }
    
    # Test normal disposal
    print("\n1. NORMAL DISPOSAL SCENARIO")
    print("-" * 50)
    econ_normal = EconParameters(base_config, mocat)
    econ_normal.calculate_cost_fn_parameters(0.95, "normal_disposal")
    
    print("Delta-v requirements for normal disposal:")
    for i, alt in enumerate(altitudes_km):
        status = "Compliant" if econ_normal.naturally_compliant_vector[i] else "Non-compliant"
        print(f"  {alt:4d}km: {econ_normal.total_deorbit_delta_v[i]:8.2f} m/s ({status})")
    
    # Test controlled re-entry
    print("\n2. CONTROLLED RE-ENTRY SCENARIO")
    print("-" * 50)
    controlled_config = base_config.copy()
    controlled_config["OPUS"]["controlled_renetries_only"] = True
    
    econ_controlled = EconParameters(controlled_config, mocat)
    econ_controlled.calculate_cost_fn_parameters(0.95, "controlled_reentry")
    
    print("Delta-v requirements for controlled re-entry (to 75km perigee):")
    for i, alt in enumerate(altitudes_km):
        status = "Compliant" if econ_controlled.naturally_compliant_vector[i] else "Non-compliant"
        print(f"  {alt:4d}km: {econ_controlled.total_deorbit_delta_v[i]:8.2f} m/s ({status})")
    
    # Comparison analysis
    print("\n3. COMPARISON ANALYSIS")
    print("-" * 50)
    print("Altitude | Normal Disposal | Controlled Re-entry | Difference | Status")
    print("---------|-----------------|---------------------|------------|--------")
    
    total_normal_cost = 0
    total_controlled_cost = 0
    
    for i, alt in enumerate(altitudes_km):
        normal_dv = econ_normal.total_deorbit_delta_v[i]
        controlled_dv = econ_controlled.total_deorbit_delta_v[i]
        diff = controlled_dv - normal_dv
        status = "Compliant" if econ_normal.naturally_compliant_vector[i] else "Non-compliant"
        
        # Calculate costs
        normal_cost = normal_dv * econ_normal.delta_v_cost[i]
        controlled_cost = controlled_dv * econ_controlled.delta_v_cost[i]
        total_normal_cost += normal_cost
        total_controlled_cost += controlled_cost
        
        print(f"  {alt:4d}km  |     {normal_dv:8.2f}     |      {controlled_dv:8.2f}      |  {diff:+6.2f}   | {status}")
    
    print(f"\n4. COST ANALYSIS")
    print("-" * 50)
    print(f"Total cost for normal disposal:     ${total_normal_cost:,.0f}")
    print(f"Total cost for controlled re-entry: ${total_controlled_cost:,.0f}")
    print(f"Additional cost for controlled re-entry: ${total_controlled_cost - total_normal_cost:,.0f}")
    print(f"Cost increase: {((total_controlled_cost - total_normal_cost) / total_normal_cost * 100):.1f}%")
    
    # Key insights
    print(f"\n5. KEY INSIGHTS")
    print("-" * 50)
    
    # Find compliant altitudes
    compliant_altitudes = [alt for i, alt in enumerate(altitudes_km) if econ_normal.naturally_compliant_vector[i]]
    non_compliant_altitudes = [alt for i, alt in enumerate(altitudes_km) if not econ_normal.naturally_compliant_vector[i]]
    
    print(f"• Compliant altitudes (no delta-v needed for normal disposal): {compliant_altitudes}")
    print(f"• Non-compliant altitudes (delta-v needed): {non_compliant_altitudes}")
    
    # Check if controlled re-entry respects compliance
    controlled_respects_compliance = all(
        econ_controlled.total_deorbit_delta_v[i] == 0 
        for i in range(len(altitudes_km)) 
        if econ_normal.naturally_compliant_vector[i]
    )
    
    if controlled_respects_compliance:
        print("✓ Controlled re-entry correctly sets delta-v to 0 for naturally compliant altitudes")
    else:
        print("✗ Controlled re-entry does not respect natural compliance")
    
    # Create visualization
    create_comparison_plot(altitudes_km, econ_normal, econ_controlled)
    
    return econ_normal, econ_controlled

def create_comparison_plot(altitudes_km, econ_normal, econ_controlled):
    """
    Create a comparison plot showing delta-v requirements.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Delta-v comparison
    x = np.arange(len(altitudes_km))
    width = 0.35
    
    # Color bars based on compliance status
    normal_colors = ['green' if econ_normal.naturally_compliant_vector[i] else 'red' 
                     for i in range(len(altitudes_km))]
    controlled_colors = ['green' if econ_normal.naturally_compliant_vector[i] else 'blue' 
                        for i in range(len(altitudes_km))]
    
    bars1 = ax1.bar(x - width/2, econ_normal.total_deorbit_delta_v, width, 
                    label='Normal Disposal', color=normal_colors, alpha=0.7)
    bars2 = ax1.bar(x + width/2, econ_controlled.total_deorbit_delta_v, width,
                    label='Controlled Re-entry', color=controlled_colors, alpha=0.7)
    
    ax1.set_xlabel('Starting Altitude (km)')
    ax1.set_ylabel('Delta-v Required (m/s)')
    ax1.set_title('Delta-v Requirements Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(altitudes_km)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{height:.0f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Cost comparison
    normal_costs = [econ_normal.total_deorbit_delta_v[i] * econ_normal.delta_v_cost[i] 
                   for i in range(len(altitudes_km))]
    controlled_costs = [econ_controlled.total_deorbit_delta_v[i] * econ_controlled.delta_v_cost[i] 
                       for i in range(len(altitudes_km))]
    
    bars3 = ax2.bar(x - width/2, normal_costs, width, 
                    label='Normal Disposal', color=normal_colors, alpha=0.7)
    bars4 = ax2.bar(x + width/2, controlled_costs, width,
                    label='Controlled Re-entry', color=controlled_colors, alpha=0.7)
    
    ax2.set_xlabel('Starting Altitude (km)')
    ax2.set_ylabel('Cost ($)')
    ax2.set_title('Cost Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(altitudes_km)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1000,
                        f'${height:,.0f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('/Users/indigobrownhall/Code/OPUS/indigo-thesis/delta-v/delta_v_scenario_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved as: delta_v_scenario_comparison.png")

def main():
    """
    Main function to run the scenario comparison.
    """
    # Test with a range of altitudes
    altitudes = [400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    
    econ_normal, econ_controlled = compare_scenarios(altitudes)
    
    return econ_normal, econ_controlled

if __name__ == "__main__":
    normal_params, controlled_params = main()
