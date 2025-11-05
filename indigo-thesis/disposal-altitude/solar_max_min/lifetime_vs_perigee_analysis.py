#!/usr/bin/env python3
"""
Orbital Lifetime Analysis for Solar Maximum (2024) and Solar Minimum (2032)
Using actual disposal lookup data for S and Su species
Starting altitude: 950 km
Disposal perigee range: 200-700 km
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from datetime import datetime
import os
import sys

# Add the OPUS utils to path
sys.path.append('/Users/indigobrownhall/Code/OPUS/OPUS/utils')
from PostMissionDisposal import get_disposal_orbits

def load_disposal_lookup(mat_file_path):
    """
    Load disposal lookup data from .mat file
    """
    data = sio.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
    lookup = data['lookup']
    
    return {
        'years': np.array(lookup.years, dtype=int).flatten(),
        'apogee_alts_km': np.array(lookup.apogee_alts_km).flatten(),
        'perigee_alts_km': np.array(lookup.perigee_alts_km).flatten(),
        'lifetimes_years': np.array(lookup.lifetimes_years),  # (ny, na, np)
        'decay_alt_km': np.array(lookup.decay_alt_km).flatten() if np.size(lookup.decay_alt_km) > 1 else np.array([lookup.decay_alt_km])
    }

def get_lifetime_for_perigee(year, apogee_km, perigee_km, satellite_type):
    """
    Get orbital lifetime for a specific perigee altitude using disposal lookup data
    """
    # Load the appropriate lookup data
    if satellite_type == "S":
        mat_path = '/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_S.mat'
    elif satellite_type == "Su":
        mat_path = '/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_Su.mat'
    else:
        raise ValueError(f"Invalid satellite type: {satellite_type}")
    
    lookup_data = load_disposal_lookup(mat_path)
    
    # Find closest year
    year_idx = int(np.argmin(np.abs(lookup_data['years'] - int(year))))
    
    # Find closest apogee altitude
    apogee_idx = int(np.argmin(np.abs(lookup_data['apogee_alts_km'] - apogee_km)))
    
    # Find closest perigee altitude
    perigee_idx = int(np.argmin(np.abs(lookup_data['perigee_alts_km'] - perigee_km)))
    
    # Get lifetime
    lifetime = lookup_data['lifetimes_years'][year_idx, apogee_idx, perigee_idx]
    
    return lifetime

def exponential_curve(x, a, b, c):
    """
    Exponential curve fitting function: y = a * exp(b * x) + c
    """
    return a * np.exp(b * x) + c

def power_curve(x, a, b, c):
    """
    Power curve fitting function: y = a * x^b + c
    """
    return a * np.power(x, b) + c

def fit_curve_to_data(perigee_data, lifetime_data):
    """
    Fit a curve to the lifetime data and return fitted curve parameters
    """
    # Remove NaN values
    valid_mask = ~np.isnan(lifetime_data)
    if np.sum(valid_mask) < 3:
        return None, None
    
    x_valid = perigee_data[valid_mask]
    y_valid = lifetime_data[valid_mask]
    
    # Try exponential fit first
    try:
        # Initial guess for exponential parameters
        p0_exp = [1e-6, 0.01, 0.1]
        popt_exp, _ = curve_fit(exponential_curve, x_valid, y_valid, p0=p0_exp, maxfev=1000)
        
        # Calculate R-squared for exponential fit
        y_pred_exp = exponential_curve(x_valid, *popt_exp)
        ss_res_exp = np.sum((y_valid - y_pred_exp) ** 2)
        ss_tot_exp = np.sum((y_valid - np.mean(y_valid)) ** 2)
        r2_exp = 1 - (ss_res_exp / ss_tot_exp)
        
        return popt_exp, 'exponential'
    except:
        # Fallback to power fit
        try:
            p0_pow = [1e-6, 2, 0.1]
            popt_pow, _ = curve_fit(power_curve, x_valid, y_valid, p0=p0_pow, maxfev=1000)
            return popt_pow, 'power'
        except:
            return None, None

def create_lifetime_plot():
    """
    Create the orbital lifetime plot for solar max/min conditions using actual disposal data
    """
    # Parameters - sample every 50km
    start_altitude = 950  # km
    perigee_range = np.arange(200, 701, 50)  # km - every 50km from 200 to 700
    
    # Calculate lifetimes for both solar conditions and satellite types
    satellite_types = ['S', 'Su']
    solar_years = {'Solar Max': 2024, 'Solar Min': 2032}
    
    results = {}
    fitted_curves = {}
    
    for sat_type in satellite_types:
        results[sat_type] = {}
        fitted_curves[sat_type] = {}
        for condition, year in solar_years.items():
            lifetimes = []
            for perigee in perigee_range:
                try:
                    lifetime = get_lifetime_for_perigee(year, start_altitude, perigee, sat_type)
                    lifetimes.append(lifetime)
                except Exception as e:
                    print(f"Error for {sat_type}, {condition}, perigee {perigee}: {e}")
                    lifetimes.append(np.nan)
            
            results[sat_type][condition] = np.array(lifetimes)
            
            # Fit curve to the data
            params, curve_type = fit_curve_to_data(perigee_range, np.array(lifetimes))
            fitted_curves[sat_type][condition] = {'params': params, 'type': curve_type}
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.style.use('dark_background')
    
    colors = {'Solar Max': 'blue', 'Solar Min': 'orange'}
    markers = {'S': 'o', 'Su': 's'}
    
    # Create fine-grained perigee range for smooth curves
    perigee_fine = np.linspace(200, 700, 1000)
    
    # Plot the data points and fitted curves
    for sat_type in satellite_types:
        for condition, year in solar_years.items():
            lifetimes = results[sat_type][condition]
            valid_mask = ~np.isnan(lifetimes)
            
            if np.any(valid_mask):
                # Plot data points
                plt.semilogy(perigee_range[valid_mask], lifetimes[valid_mask], 
                           color=colors[condition], marker=markers[sat_type],
                           linewidth=0, markersize=8, 
                           label=f'{sat_type} - {condition} ({year})', 
                           markerfacecolor='none', markeredgewidth=2)
                
                # Plot fitted curve
                curve_info = fitted_curves[sat_type][condition]
                if curve_info['params'] is not None:
                    params = curve_info['params']
                    curve_type = curve_info['type']
                    
                    if curve_type == 'exponential':
                        curve_lifetimes = exponential_curve(perigee_fine, *params)
                    else:  # power
                        curve_lifetimes = power_curve(perigee_fine, *params)
                    
                    # Only plot curve where it makes physical sense (positive lifetimes)
                    valid_curve_mask = curve_lifetimes > 0.01
                    plt.semilogy(perigee_fine[valid_curve_mask], curve_lifetimes[valid_curve_mask], 
                               color=colors[condition], linestyle='-', linewidth=2, alpha=0.8)
    
    # Find and plot vertical lines for 5-year lifetime disposal perigees
    disposal_perigees_5yr = {}
    
    for sat_type in satellite_types:
        disposal_perigees_5yr[sat_type] = {}
        for condition, year in solar_years.items():
            lifetimes = results[sat_type][condition]
            valid_mask = ~np.isnan(lifetimes)
            
            if np.any(valid_mask):
                valid_lifetimes = lifetimes[valid_mask]
                valid_perigees = perigee_range[valid_mask]
                
                # Find closest point to 5 years
                idx_5 = np.argmin(np.abs(valid_lifetimes - 5))
                disposal_perigee_5yr = valid_perigees[idx_5]
                disposal_perigees_5yr[sat_type][condition] = disposal_perigee_5yr
                
                # Plot vertical line for 5-year disposal perigee
                plt.axvline(x=disposal_perigee_5yr, color=colors[condition], 
                           linestyle=':', alpha=0.7, linewidth=2)
    
    # Add horizontal reference line for 5-year lifetime
    plt.axhline(y=5, color='white', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Highlight specific points (5-year lifetimes)
    for sat_type in satellite_types:
        for condition, year in solar_years.items():
            lifetimes = results[sat_type][condition]
            valid_mask = ~np.isnan(lifetimes)
            
            if np.any(valid_mask):
                # Find closest point to 5 years
                valid_lifetimes = lifetimes[valid_mask]
                valid_perigees = perigee_range[valid_mask]
                
                if len(valid_lifetimes) > 0:
                    idx_5 = np.argmin(np.abs(valid_lifetimes - 5))
                    
                    # Highlight 5-year point
                    plt.plot(valid_perigees[idx_5], valid_lifetimes[idx_5], 
                             color=colors[condition], marker=markers[sat_type],
                             markersize=10, markerfacecolor=colors[condition], 
                             markeredgecolor='white', markeredgewidth=2)
    
    # Customize the plot
    plt.xlabel('Disposal Perigee (km)', fontsize=14, fontweight='bold')
    plt.ylabel('Orbital Lifetime (years, log scale)', fontsize=14, fontweight='bold')
    plt.title('Lifetime vs Perigee Altitude — Solar Max (2024) vs Solar Min (2032)', fontsize=16, fontweight='bold')
    
    # Set axis limits and ticks
    plt.xlim(200, 700)
    plt.ylim(0.1, 100)
    
    # Customize ticks
    plt.xticks(np.arange(200, 701, 50))
    plt.yticks([0.1, 1, 10, 100], ['0.1', '1', '10', '100'])
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.grid(True, which='minor', alpha=0.2, linestyle='-', linewidth=0.3)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', marker='o', linestyle='-', linewidth=2, 
               markersize=6, markerfacecolor='none', markeredgewidth=2, label='S - Solar Max (2024)'),
        Line2D([0], [0], color='blue', marker='s', linestyle='-', linewidth=2, 
               markersize=6, markerfacecolor='none', markeredgewidth=2, label='Su - Solar Max (2024)'),
        Line2D([0], [0], color='orange', marker='o', linestyle='-', linewidth=2, 
               markersize=6, markerfacecolor='none', markeredgewidth=2, label='S - Solar Min (2032)'),
        Line2D([0], [0], color='orange', marker='s', linestyle='-', linewidth=2, 
               markersize=6, markerfacecolor='none', markeredgewidth=2, label='Su - Solar Min (2032)'),
        Line2D([0], [0], color='white', linestyle='--', linewidth=1.5, 
               label='5-year lifetime'),
        Line2D([0], [0], color='blue', linestyle=':', linewidth=2, 
               label='5-yr disposal perigee (Solar Max)'),
        Line2D([0], [0], color='orange', linestyle=':', linewidth=2, 
               label='5-yr disposal perigee (Solar Min)')
    ]
    
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = '/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/solar_max_min/lifetime_vs_perigee_solar_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
    
    print(f"Plot saved to: {output_path}")
    
    # Show plot info
    print(f"\nAnalysis Results:")
    print(f"Starting altitude: {start_altitude} km")
    print(f"Perigee range: {perigee_range[0]} - {perigee_range[-1]} km (sampled every 50km)")
    
    print(f"\n5-Year Disposal Perigee Altitudes:")
    for sat_type in satellite_types:
        print(f"\n{sat_type} Satellites:")
        for condition, year in solar_years.items():
            if sat_type in disposal_perigees_5yr and condition in disposal_perigees_5yr[sat_type]:
                disposal_perigee = disposal_perigees_5yr[sat_type][condition]
                print(f"  {condition} ({year}): {disposal_perigee:.0f} km")
    
    print(f"\nCurve Fitting Results:")
    for sat_type in satellite_types:
        print(f"\n{sat_type} Satellites:")
        for condition, year in solar_years.items():
            curve_info = fitted_curves[sat_type][condition]
            if curve_info['params'] is not None:
                params = curve_info['params']
                curve_type = curve_info['type']
                print(f"  {condition} ({year}): {curve_type} curve fitted")
                print(f"    Parameters: {params}")
            else:
                print(f"  {condition} ({year}): No curve fitted (insufficient data)")
    
    return perigee_range, results, disposal_perigees_5yr

def main():
    """
    Main function to run the orbital lifetime analysis
    """
    print("Orbital Lifetime Analysis for Solar Maximum/Minimum Conditions")
    print("Using actual disposal lookup data for S and Su species")
    print("=" * 70)
    
    # Create the plot
    perigee_range, results, disposal_perigees_5yr = create_lifetime_plot()
    
    print(f"\nAnalysis complete!")
    print(f"Generated plot comparing solar maximum (2024) and solar minimum (2032) conditions")
    print(f"Using actual disposal lookup data for S and Su satellite types")
    print(f"Curves fitted to data points sampled every 50km")
    print(f"Vertical dotted lines show disposal perigee altitudes for 5-year lifetimes")

if __name__ == "__main__":
    main()
