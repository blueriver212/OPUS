#!/usr/bin/env python3
"""
Delta-v Heatmap Analysis: 25-year Lifetime vs Direct Re-entry

This script creates a heatmap showing delta-v requirements for satellites at different
altitudes (600-1500km) for two scenarios:
1. Achieving 25-year orbital lifetime compliance
2. Direct re-entry to 70km perigee

The analysis uses atmospheric density models and orbital mechanics to determine
the highest compliant altitude for 25-year lifetime and calculates the required
delta-v for both scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyssem.utils.drag.drag import densityexp
import pandas as pd
import sys
import os

# Add the OPUS directory to the path to import EconParameters
sys.path.append('/Users/indigobrownhall/Code/OPUS')

# Earth's gravitational parameter (m^3/s^2)
MU_EARTH = 3.986004418e14

# Earth's radius (m)
R_EARTH = 6.371e6

# Remove the custom density function - we'll use the real pyssem one

def calculate_orbital_lifetime(altitude_km, ballistic_coefficient=0.0172, disposal_time=25):
    """
    Calculate orbital lifetime for a satellite at given altitude using the same method
    as EconParameters.py.
    
    Parameters:
    altitude_km (float): Altitude in km
    ballistic_coefficient (float): Ballistic coefficient (m^2/kg)
    disposal_time (float): Required disposal time in years
    
    Returns:
    tuple: (is_compliant, lifetime_years, cumulative_residence_time)
    """
    # Convert altitude to radius
    r = R_EARTH + altitude_km * 1000
    
    # Atmospheric density using the same method as EconParameters.py
    rho = densityexp(altitude_km)
    
    # Calculate decay rate using the same formula as EconParameters.py
    beta = ballistic_coefficient  # m^2/kg
    rvel_current_D = -rho * beta * np.sqrt(MU_EARTH * r) * (24 * 3600 * 365.25)
    
    # Convert to altitude change rate (assuming shell height of 50km)
    Dhl = 50e3  # shell height in meters
    shell_marginal_decay_rate = -rvel_current_D / Dhl
    shell_marginal_residence_time = 1 / shell_marginal_decay_rate
    
    # For a single shell, cumulative residence time equals marginal residence time
    cumulative_residence_time = shell_marginal_residence_time
    
    # Check if compliant (cumulative residence time >= disposal time)
    is_compliant = cumulative_residence_time >= disposal_time
    
    return is_compliant, cumulative_residence_time, shell_marginal_residence_time

def find_25_year_compliant_altitude():
    """
    Find the highest altitude that provides 25-year orbital lifetime.
    Uses binary search approach.
    """
    # Search range
    min_alt = 200  # km
    max_alt = 2000  # km
    target_lifetime = 25  # years
    tolerance = 0.1  # years
    
    while max_alt - min_alt > 1:
        mid_alt = (min_alt + max_alt) / 2
        is_compliant, lifetime, _ = calculate_orbital_lifetime(mid_alt)
        
        if is_compliant:
            min_alt = mid_alt
        else:
            max_alt = mid_alt
    
    return min_alt

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
    """
    # Convert to meters
    r1 = R_EARTH + initial_altitude_km * 1000  # Initial orbit radius
    r2 = R_EARTH + target_perigee_km * 1000    # Target perigee radius
    
    # Calculate Hohmann transfer delta-v
    delta_v1, delta_v2 = calculate_hohmann_delta_v(r1, r2)
    
    # Total delta-v (sum of both burns) - take absolute value for deorbit
    total_delta_v = abs(delta_v1) + abs(delta_v2)
    
    return total_delta_v

def create_heatmap_data():
    """
    Create data for the heatmap showing delta-v requirements for different scenarios.
    """
    # Altitude range: 600-1500km in 50km increments
    altitudes = np.arange(600, 1550, 50)
    
    # Find 25-year compliant altitude
    compliant_altitude = find_25_year_compliant_altitude()
    print(f"25-year compliant altitude: {compliant_altitude:.1f} km")
    
    # Initialize data arrays
    n_altitudes = len(altitudes)
    lifetime_delta_v = np.zeros(n_altitudes)
    reentry_delta_v = np.zeros(n_altitudes)
    lifetimes = np.zeros(n_altitudes)
    
    # Calculate for each altitude
    for i, alt in enumerate(altitudes):
        # Calculate actual lifetime using EconParameters method
        is_compliant, lifetime, _ = calculate_orbital_lifetime(alt)
        lifetimes[i] = lifetime
        
        # Delta-v for 25-year compliance (if needed)
        if not is_compliant:
            # Need to raise altitude to compliant level
            lifetime_delta_v[i] = calculate_deorbit_delta_v(alt, compliant_altitude)
        else:
            # Already compliant, no delta-v needed
            lifetime_delta_v[i] = 0
        
        # Delta-v for direct re-entry to 70km
        reentry_delta_v[i] = calculate_deorbit_delta_v(alt, 70)
    
    return altitudes, lifetime_delta_v, reentry_delta_v, lifetimes, compliant_altitude

def create_heatmap_visualization(altitudes, lifetime_delta_v, reentry_delta_v, lifetimes, compliant_altitude):
    """
    Create comprehensive heatmap visualization.
    """
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Delta-v for 25-year compliance
    im1 = ax1.imshow(lifetime_delta_v.reshape(1, -1), aspect='auto', cmap='Reds', 
                     extent=[altitudes[0], altitudes[-1], 0, 1])
    ax1.set_xlabel('Starting Altitude (km)')
    ax1.set_title('Delta-v for 25-year Compliance (m/s)')
    ax1.set_yticks([])
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Delta-v (m/s)')
    
    # Add value annotations
    for i, (alt, dv) in enumerate(zip(altitudes, lifetime_delta_v)):
        if dv > 0:
            ax1.text(alt, 0.5, f'{dv:.0f}', ha='center', va='center', 
                    color='white' if dv > np.max(lifetime_delta_v)/2 else 'black')
    
    # 2. Delta-v for direct re-entry
    im2 = ax2.imshow(reentry_delta_v.reshape(1, -1), aspect='auto', cmap='Blues',
                     extent=[altitudes[0], altitudes[-1], 0, 1])
    ax2.set_xlabel('Starting Altitude (km)')
    ax2.set_title('Delta-v for Direct Re-entry to 70km (m/s)')
    ax2.set_yticks([])
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Delta-v (m/s)')
    
    # Add value annotations
    for i, (alt, dv) in enumerate(zip(altitudes, reentry_delta_v)):
        ax2.text(alt, 0.5, f'{dv:.0f}', ha='center', va='center',
                color='white' if dv > np.max(reentry_delta_v)/2 else 'black')
    
    # 3. Natural lifetime vs altitude
    ax3.plot(altitudes, lifetimes, 'g-', linewidth=2, marker='o', markersize=4)
    ax3.axhline(y=25, color='r', linestyle='--', alpha=0.7, label='25-year requirement')
    ax3.axvline(x=compliant_altitude, color='r', linestyle='--', alpha=0.7, 
               label=f'Compliant altitude ({compliant_altitude:.1f} km)')
    ax3.set_xlabel('Starting Altitude (km)')
    ax3.set_ylabel('Natural Lifetime (years)')
    ax3.set_title('Natural Orbital Lifetime vs Altitude')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Comparison: Lifetime vs Re-entry delta-v
    ax4.plot(altitudes, lifetime_delta_v, 'r-', linewidth=2, marker='o', 
             markersize=4, label='25-year Compliance')
    ax4.plot(altitudes, reentry_delta_v, 'b-', linewidth=2, marker='s', 
             markersize=4, label='Direct Re-entry')
    ax4.set_xlabel('Starting Altitude (km)')
    ax4.set_ylabel('Delta-v Required (m/s)')
    ax4.set_title('Delta-v Comparison: Compliance vs Re-entry')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('/Users/indigobrownhall/Code/OPUS/indigo-thesis/delta-v/lifetime_vs_reentry_heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_summary_table(altitudes, lifetime_delta_v, reentry_delta_v, lifetimes, compliant_altitude):
    """
    Create a summary table of the results.
    """
    # Create DataFrame
    df = pd.DataFrame({
        'Starting_Altitude_km': altitudes,
        'Natural_Lifetime_years': lifetimes,
        'Compliant_25yr_DeltaV_ms': lifetime_delta_v,
        'Direct_Reentry_DeltaV_ms': reentry_delta_v,
        'DeltaV_Difference_ms': reentry_delta_v - lifetime_delta_v
    })
    
    # Add compliance status based on actual compliance check
    compliance_status = []
    for i, alt in enumerate(altitudes):
        is_compliant, _, _ = calculate_orbital_lifetime(alt)
        compliance_status.append('Compliant' if is_compliant else 'Non-compliant')
    
    df['Compliance_Status'] = compliance_status
    
    print("\n" + "="*100)
    print("SUMMARY TABLE: DELTA-V REQUIREMENTS BY ALTITUDE")
    print("="*100)
    print(df.to_string(index=False, float_format='%.1f'))
    
    # Save to CSV
    df.to_csv('/Users/indigobrownhall/Code/OPUS/indigo-thesis/delta-v/delta_v_summary.csv', index=False)
    print(f"\nSummary table saved as: delta_v_summary.csv")
    
    return df

def main():
    """
    Main analysis function.
    """
    print("="*80)
    print("DELTA-V HEATMAP ANALYSIS: 25-YEAR LIFETIME vs DIRECT RE-ENTRY")
    print("="*80)
    
    # Create heatmap data
    altitudes, lifetime_delta_v, reentry_delta_v, lifetimes, compliant_altitude = create_heatmap_data()
    
    # Create visualizations
    fig = create_heatmap_visualization(altitudes, lifetime_delta_v, reentry_delta_v, 
                                     lifetimes, compliant_altitude)
    
    # Create summary table
    df = create_summary_table(altitudes, lifetime_delta_v, reentry_delta_v, 
                            lifetimes, compliant_altitude)
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Find altitudes that are naturally compliant
    compliant_altitudes = []
    non_compliant_altitudes = []
    
    for alt in altitudes:
        is_compliant, _, _ = calculate_orbital_lifetime(alt)
        if is_compliant:
            compliant_altitudes.append(alt)
        else:
            non_compliant_altitudes.append(alt)
    
    print(f"• 25-year compliant altitude threshold: {compliant_altitude:.1f} km")
    print(f"• Naturally compliant altitudes: {compliant_altitudes}")
    print(f"• Non-compliant altitudes requiring delta-v: {non_compliant_altitudes}")
    
    # Delta-v statistics
    print(f"\n• Maximum delta-v for 25-year compliance: {np.max(lifetime_delta_v):.1f} m/s")
    print(f"• Maximum delta-v for direct re-entry: {np.max(reentry_delta_v):.1f} m/s")
    print(f"• Average delta-v difference (re-entry - compliance): {np.mean(reentry_delta_v - lifetime_delta_v):.1f} m/s")
    
    # Cost implications (assuming $1000/m/s)
    cost_per_ms = 1000
    max_compliance_cost = np.max(lifetime_delta_v) * cost_per_ms
    max_reentry_cost = np.max(reentry_delta_v) * cost_per_ms
    
    print(f"\n• Maximum cost for 25-year compliance: ${max_compliance_cost:,.0f}")
    print(f"• Maximum cost for direct re-entry: ${max_reentry_cost:,.0f}")
    print(f"• Cost difference: ${max_reentry_cost - max_compliance_cost:,.0f}")
    
    return df

if __name__ == "__main__":
    results_df = main()
