#!/usr/bin/env python3
"""
Simplified Delta-v Analysis using EconParameters method

This script creates a heatmap showing delta-v requirements for satellites at different
altitudes (600-1500km) for two scenarios:
1. Achieving 25-year orbital lifetime compliance
2. Direct re-entry to 70km perigee

Uses the exact same method as EconParameters.py for lifetime calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyssem.utils.drag.drag import densityexp
import pandas as pd

# Earth's gravitational parameter (m^3/s^2)
MU_EARTH = 3.986004418e14

# Earth's radius (m)
R_EARTH = 6.371e6

def calculate_shell_residence_times(altitudes_km, disposal_time=25, ballistic_coefficient=0.0172):
    """
    Calculate residence times for each altitude using the EconParameters method.
    
    Parameters:
    altitudes_km (array): Array of altitudes in km
    disposal_time (float): Required disposal time in years
    ballistic_coefficient (float): Ballistic coefficient (m^2/kg)
    
    Returns:
    tuple: (is_compliant_array, residence_times, k_star)
    """
    n_altitudes = len(altitudes_km)
    
    # Initialize arrays
    shell_marginal_decay_rates = np.zeros(n_altitudes)
    shell_marginal_residence_times = np.zeros(n_altitudes)
    shell_cumulative_residence_times = np.zeros(n_altitudes)
    
    # Calculate residence times for each altitude using EconParameters method
    for k, altitude_km in enumerate(altitudes_km):
        # Convert altitude to radius
        R0_km = altitude_km
        R0 = R_EARTH + altitude_km * 1000  # radius in meters
        
        # Atmospheric density using pyssem
        rhok = densityexp(R0_km)
        
        # Calculate decay rate using EconParameters formula
        beta = ballistic_coefficient
        rvel_current_D = -rhok * beta * np.sqrt(MU_EARTH * R0) * (24 * 3600 * 365.25)
        
        # Convert to altitude change rate (assuming shell height of 50km)
        Dhl = 50e3  # shell height in meters
        shell_marginal_decay_rates[k] = -rvel_current_D / Dhl
        shell_marginal_residence_times[k] = 1 / shell_marginal_decay_rates[k]
    
    # Calculate cumulative residence times
    shell_cumulative_residence_times = np.cumsum(shell_marginal_residence_times)
    
    # Find k_star - the highest altitude that meets disposal time requirement
    indices = np.where(shell_cumulative_residence_times <= disposal_time)[0]
    k_star = max(indices) if len(indices) > 0 else 0
    
    # Determine compliance for each altitude
    is_compliant = np.zeros(n_altitudes, dtype=bool)
    for k in range(n_altitudes):
        if k <= k_star:
            is_compliant[k] = True
        else:
            is_compliant[k] = False
    
    return is_compliant, shell_cumulative_residence_times, k_star

def calculate_hohmann_delta_v(r1, r2):
    """
    Calculate delta-v for Hohmann transfer between two circular orbits.
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

def calculate_orbit_raise_delta_v(initial_altitude_km, target_altitude_km):
    """
    Calculate total delta-v required to raise orbit from initial altitude to target altitude.
    This is the same as deorbit but in reverse - we're going to a higher altitude.
    """
    # Convert to meters
    r1 = R_EARTH + initial_altitude_km * 1000  # Initial orbit radius
    r2 = R_EARTH + target_altitude_km * 1000   # Target orbit radius
    
    # Calculate Hohmann transfer delta-v
    delta_v1, delta_v2 = calculate_hohmann_delta_v(r1, r2)
    
    # For orbit raising, we need to add velocity, so we take the positive values
    # delta_v1 should be positive (we're going to higher orbit)
    # delta_v2 should be positive (we're circularizing at higher altitude)
    total_delta_v = abs(delta_v1) + abs(delta_v2)
    
    return total_delta_v

def main():
    """
    Main analysis function.
    """
    print("="*80)
    print("DELTA-V HEATMAP ANALYSIS: 25-YEAR LIFETIME vs DIRECT RE-ENTRY")
    print("="*80)
    
    # Altitude range: 600-1500km in 50km increments
    altitudes = np.arange(600, 1550, 50)
    print(f"Analyzing altitudes: {altitudes} km")
    
    # Calculate compliance using EconParameters method
    is_compliant, residence_times, k_star = calculate_shell_residence_times(altitudes)
    
    # Find the highest compliant altitude
    compliant_altitude = altitudes[k_star] if k_star < len(altitudes) else altitudes[-1]
    
    print(f"\n25-year compliant altitude threshold: {compliant_altitude} km")
    print(f"Compliant altitudes: {altitudes[is_compliant].tolist()}")
    print(f"Non-compliant altitudes: {altitudes[~is_compliant].tolist()}")
    
    # Calculate delta-v requirements
    n_altitudes = len(altitudes)
    lifetime_delta_v = np.zeros(n_altitudes)
    reentry_delta_v = np.zeros(n_altitudes)
    
    for i, alt in enumerate(altitudes):
        # Delta-v for 25-year compliance (if needed)
        if not is_compliant[i]:
            # Need to raise altitude to compliant level - this requires delta-v to go UP
            # We need to calculate delta-v from current altitude to compliant altitude
            lifetime_delta_v[i] = calculate_orbit_raise_delta_v(alt, compliant_altitude)
        else:
            # Already compliant, no delta-v needed
            lifetime_delta_v[i] = 0
        
        # Delta-v for direct re-entry to 70km
        reentry_delta_v[i] = calculate_deorbit_delta_v(alt, 70)
    
    # Create results DataFrame
    df = pd.DataFrame({
        'Starting_Altitude_km': altitudes,
        'Residence_Time_years': residence_times,
        'Compliant_25yr_DeltaV_ms': lifetime_delta_v,
        'Direct_Reentry_DeltaV_ms': reentry_delta_v,
        'DeltaV_Difference_ms': reentry_delta_v - lifetime_delta_v,
        'Compliance_Status': ['Compliant' if c else 'Non-compliant' for c in is_compliant]
    })
    
    print("\n" + "="*100)
    print("SUMMARY TABLE: DELTA-V REQUIREMENTS BY ALTITUDE")
    print("="*100)
    print(df.to_string(index=False, float_format='%.1f'))
    
    # Save to CSV
    df.to_csv('/Users/indigobrownhall/Code/OPUS/indigo-thesis/delta-v/simple_delta_v_summary.csv', index=False)
    print(f"\nSummary table saved as: simple_delta_v_summary.csv")
    
    # Create visualization
    create_heatmap_visualization(altitudes, lifetime_delta_v, reentry_delta_v, 
                               residence_times, is_compliant, compliant_altitude)
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    print(f"• 25-year compliant altitude threshold: {compliant_altitude} km")
    print(f"• Naturally compliant altitudes: {altitudes[is_compliant].tolist()}")
    print(f"• Non-compliant altitudes requiring delta-v: {altitudes[~is_compliant].tolist()}")
    
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

def create_heatmap_visualization(altitudes, lifetime_delta_v, reentry_delta_v, 
                               residence_times, is_compliant, compliant_altitude):
    """
    Create comprehensive heatmap visualization.
    """
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Delta-v for 25-year compliance
    colors1 = ['green' if c else 'red' for c in is_compliant]
    bars1 = ax1.bar(range(len(altitudes)), lifetime_delta_v, color=colors1, alpha=0.7)
    ax1.set_xlabel('Starting Altitude (km)')
    ax1.set_ylabel('Delta-v for 25-year Compliance (m/s)')
    ax1.set_title('Delta-v for 25-year Compliance')
    ax1.set_xticks(range(len(altitudes)))
    ax1.set_xticklabels(altitudes, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (bar, dv) in enumerate(zip(bars1, lifetime_delta_v)):
        if dv > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                    f'{dv:.0f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Delta-v for direct re-entry
    bars2 = ax2.bar(range(len(altitudes)), reentry_delta_v, color='blue', alpha=0.7)
    ax2.set_xlabel('Starting Altitude (km)')
    ax2.set_ylabel('Delta-v for Direct Re-entry (m/s)')
    ax2.set_title('Delta-v for Direct Re-entry to 70km')
    ax2.set_xticks(range(len(altitudes)))
    ax2.set_xticklabels(altitudes, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (bar, dv) in enumerate(zip(bars2, reentry_delta_v)):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'{dv:.0f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Residence time vs altitude
    colors3 = ['green' if c else 'red' for c in is_compliant]
    ax3.scatter(altitudes, residence_times, c=colors3, s=50, alpha=0.7)
    ax3.axhline(y=25, color='black', linestyle='--', alpha=0.7, label='25-year requirement')
    ax3.axvline(x=compliant_altitude, color='black', linestyle='--', alpha=0.7, 
               label=f'Compliant altitude ({compliant_altitude} km)')
    ax3.set_xlabel('Starting Altitude (km)')
    ax3.set_ylabel('Residence Time (years)')
    ax3.set_title('Natural Residence Time vs Altitude')
    ax3.set_yscale('log')
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
    plt.savefig('/Users/indigobrownhall/Code/OPUS/indigo-thesis/delta-v/simple_lifetime_heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved as: simple_lifetime_heatmap.png")

if __name__ == "__main__":
    results_df = main()
