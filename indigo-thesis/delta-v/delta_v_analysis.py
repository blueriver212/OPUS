#!/usr/bin/env python3
"""
Delta-v Analysis for Satellite Disposal Orbits

This script calculates the delta-v required for a satellite at 800km altitude
to reach different perigee altitudes (70km and 500km) using Hohmann transfer equations.

The analysis is based on the delta-v calculations used in the OPUS EconParameters class.
"""

import numpy as np
import matplotlib.pyplot as plt

# Earth's gravitational parameter (m^3/s^2)
MU_EARTH = 3.986004418e14

# Earth's radius (m)
R_EARTH = 6.371e6

def calculate_hohmann_delta_v(r1, r2):
    """
    Calculate delta-v for Hohmann transfer between two circular orbits.
    
    Parameters:
    r1 (float): Initial orbit radius (m)
    r2 (float): Final orbit radius (m)
    
    Returns:
    tuple: (delta_v1, delta_v2) - delta-v for first and second burns (m/s)
    """
    # Ensure r1 > r2 for deorbit (r1 is higher altitude)
    if r1 < r2:
        r1, r2 = r2, r1
    
    # Orbital velocities
    v1 = np.sqrt(MU_EARTH / r1)  # Initial circular orbit velocity
    v2 = np.sqrt(MU_EARTH / r2)  # Final circular orbit velocity
    
    # Transfer orbit velocities at perigee and apogee
    v_transfer_perigee = np.sqrt(MU_EARTH * (2/r2 - 2/(r1 + r2)))
    v_transfer_apogee = np.sqrt(MU_EARTH * (2/r1 - 2/(r1 + r2)))
    
    # Delta-v for first burn (circularize at apogee of transfer orbit)
    delta_v1 = v_transfer_apogee - v1
    
    # Delta-v for second burn (circularize at perigee of transfer orbit)
    delta_v2 = v2 - v_transfer_perigee
    
    return delta_v1, delta_v2

def calculate_deorbit_delta_v(initial_altitude_km, target_perigee_km):
    """
    Calculate total delta-v required to deorbit from initial altitude to target perigee.
    
    This uses the same approach as in EconParameters.py but for specific altitudes.
    
    Parameters:
    initial_altitude_km (float): Initial altitude in km
    target_perigee_km (float): Target perigee altitude in km
    
    Returns:
    float: Total delta-v required (m/s)
    """
    # Convert to meters
    r1 = R_EARTH + initial_altitude_km * 1000  # Initial orbit radius
    r2 = R_EARTH + target_perigee_km * 1000    # Target perigee radius
    
    # Calculate Hohmann transfer delta-v
    delta_v1, delta_v2 = calculate_hohmann_delta_v(r1, r2)
    
    # Total delta-v (sum of both burns) - take absolute value for deorbit
    total_delta_v = abs(delta_v1) + abs(delta_v2)
    
    return total_delta_v

def main():
    """
    Main analysis function for 800km satellite deorbit scenarios.
    """
    print("=" * 60)
    print("DELTA-V ANALYSIS: 800km SATELLITE DEORBIT SCENARIOS")
    print("=" * 60)
    
    # Initial conditions
    initial_altitude = 800  # km
    target_perigees = [70, 500]  # km
    
    print(f"\nInitial satellite altitude: {initial_altitude} km")
    print(f"Target perigee altitudes: {target_perigees} km")
    
    # Calculate delta-v for each scenario
    results = {}
    
    for target_perigee in target_perigees:
        delta_v = calculate_deorbit_delta_v(initial_altitude, target_perigee)
        results[target_perigee] = delta_v
        
        print(f"\nTarget perigee: {target_perigee} km")
        print(f"Required delta-v: {delta_v:.2f} m/s")
        print(f"Required delta-v: {delta_v/1000:.3f} km/s")
    
    # Calculate the difference
    delta_v_70km = results[70]
    delta_v_500km = results[500]
    difference = delta_v_70km - delta_v_500km
    
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Delta-v to 70km perigee:  {delta_v_70km:.2f} m/s ({delta_v_70km/1000:.3f} km/s)")
    print(f"Delta-v to 500km perigee: {delta_v_500km:.2f} m/s ({delta_v_500km/1000:.3f} km/s)")
    print(f"Difference (70km - 500km): {difference:.2f} m/s ({difference/1000:.3f} km/s)")
    print(f"Percentage difference: {((delta_v_70km - delta_v_500km) / delta_v_500km * 100):.1f}%")
    
    # Additional analysis: Show the breakdown for each scenario
    print("\n" + "=" * 60)
    print("DETAILED BREAKDOWN")
    print("=" * 60)
    
    for target_perigee in target_perigees:
        r1 = R_EARTH + initial_altitude * 1000
        r2 = R_EARTH + target_perigee * 1000
        
        delta_v1, delta_v2 = calculate_hohmann_delta_v(r1, r2)
        
        print(f"\nTarget perigee: {target_perigee} km")
        print(f"  First burn (circularize at apogee): {abs(delta_v1):.2f} m/s")
        print(f"  Second burn (circularize at perigee): {abs(delta_v2):.2f} m/s")
        print(f"  Total delta-v: {abs(delta_v1) + abs(delta_v2):.2f} m/s")
    
    # Create a visualization
    create_delta_v_plot(initial_altitude, target_perigees, results)
    
    return results

def create_delta_v_plot(initial_altitude, target_perigees, results):
    """
    Create a visualization of the delta-v requirements.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Bar chart of delta-v requirements
    perigees = [str(p) for p in target_perigees]
    delta_vs = [results[p] for p in target_perigees]
    
    bars = ax1.bar(perigees, delta_vs, color=['red', 'blue'], alpha=0.7)
    ax1.set_xlabel('Target Perigee Altitude (km)')
    ax1.set_ylabel('Delta-v Required (m/s)')
    ax1.set_title(f'Delta-v Required from {initial_altitude}km to Different Perigees')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, delta_v in zip(bars, delta_vs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{delta_v:.1f} m/s', ha='center', va='bottom')
    
    # Plot 2: Orbit visualization
    ax2.set_aspect('equal')
    
    # Earth
    earth_circle = plt.Circle((0, 0), R_EARTH/1e6, color='blue', alpha=0.3, label='Earth')
    ax2.add_patch(earth_circle)
    
    # Initial orbit (800km)
    initial_radius = (R_EARTH + initial_altitude * 1000) / 1e6
    initial_circle = plt.Circle((0, 0), initial_radius, fill=False, color='green', 
                               linestyle='-', linewidth=2, label=f'Initial Orbit ({initial_altitude}km)')
    ax2.add_patch(initial_circle)
    
    # Target perigees
    colors = ['red', 'blue']
    for i, perigee in enumerate(target_perigees):
        target_radius = (R_EARTH + perigee * 1000) / 1e6
        target_circle = plt.Circle((0, 0), target_radius, fill=False, color=colors[i], 
                                  linestyle='--', linewidth=2, label=f'Target Perigee ({perigee}km)')
        ax2.add_patch(target_circle)
    
    ax2.set_xlim(-8000, 8000)
    ax2.set_ylim(-8000, 8000)
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Distance (km)')
    ax2.set_title('Orbit Visualization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/indigobrownhall/Code/OPUS/indigo-thesis/delta-v/delta_v_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved as: delta_v_analysis.png")

if __name__ == "__main__":
    results = main()
