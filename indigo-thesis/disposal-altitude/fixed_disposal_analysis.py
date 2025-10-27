#!/usr/bin/env python3
"""
Fixed disposal analysis using interpolation instead of log-quadratic fitting
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sys
import os

def load_disposal_lookup(mat_file_path):
    """Load disposal lookup data from .mat file"""
    data = sio.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
    lookup = data['lookup']
    
    return {
        'years': np.array(lookup.years, dtype=int).flatten(),
        'apogee_alts_km': np.array(lookup.apogee_alts_km).flatten(),
        'perigee_alts_km': np.array(lookup.perigee_alts_km).flatten(),
        'lifetimes_years': np.array(lookup.lifetimes_years),
        'decay_alt_km': np.array(lookup.decay_alt_km).item() if np.size(lookup.decay_alt_km)==1 else np.array(lookup.decay_alt_km)
    }

def fixed_get_disposal_orbits(year, apogees_km, satellite_type, pmd_lifetime=5.0):
    """
    Fixed version that works with the actual .mat file structure using interpolation
    """
    
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

def create_disposal_altitude_plot_fixed():
    """Create plot showing perigee heights for varying lifetimes using fixed function"""
    
    # Load the disposal lookup data for both S and Su
    s_data = load_disposal_lookup('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_S.mat')
    su_data = load_disposal_lookup('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_Su.mat')
    
    # Define lifetime range (1-25 years)
    lifetimes = np.arange(1, 26, 1)
    
    # Define apogee heights to test (using available range from data)
    apogee_heights = s_data['apogee_alts_km']  # [400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors for different apogee heights
    colors = plt.cm.viridis(np.linspace(0, 1, len(apogee_heights)))
    
    # Plot for S satellites
    ax1.set_title('S Satellites - Perigee Height vs Lifetime', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Lifetime (years)', fontsize=12)
    ax1.set_ylabel('Required Perigee Height (km)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    for i, apogee in enumerate(apogee_heights):
        perigee_heights = []
        for lifetime in lifetimes:
            # Use the fixed function to find required perigee
            perigee = fixed_get_disposal_orbits(2024, apogee, "S", pmd_lifetime=lifetime)
            perigee_heights.append(perigee[0] if not np.isnan(perigee[0]) else np.nan)
        
        # Plot only if we have valid data
        valid_mask = ~np.isnan(perigee_heights)
        if np.any(valid_mask):
            ax1.plot(np.array(lifetimes)[valid_mask], np.array(perigee_heights)[valid_mask], 
                    'o-', color=colors[i], label=f'Apogee {apogee} km', markersize=4)
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.set_xlim(1, 25)
    
    # Plot for Su satellites
    ax2.set_title('Su Satellites - Perigee Height vs Lifetime', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Lifetime (years)', fontsize=12)
    ax2.set_ylabel('Required Perigee Height (km)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    for i, apogee in enumerate(apogee_heights):
        perigee_heights = []
        for lifetime in lifetimes:
            # Use the fixed function to find required perigee
            perigee = fixed_get_disposal_orbits(2024, apogee, "Su", pmd_lifetime=lifetime)
            perigee_heights.append(perigee[0] if not np.isnan(perigee[0]) else np.nan)
        
        # Plot only if we have valid data
        valid_mask = ~np.isnan(perigee_heights)
        if np.any(valid_mask):
            ax2.plot(np.array(lifetimes)[valid_mask], np.array(perigee_heights)[valid_mask], 
                    'o-', color=colors[i], label=f'Apogee {apogee} km', markersize=4)
    
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.set_xlim(1, 25)
    
    plt.tight_layout()
    plt.savefig('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_altitude_analysis_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_heatmap_plot_fixed():
    """Create heatmap showing perigee heights for different apogee/lifetime combinations using fixed function"""
    
    # Load the disposal lookup data
    s_data = load_disposal_lookup('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_S.mat')
    su_data = load_disposal_lookup('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_Su.mat')
    
    # Define ranges
    lifetimes = np.arange(1, 26, 1)
    apogee_heights = s_data['apogee_alts_km']
    
    # Create heatmap data
    s_perigee_matrix = np.full((len(apogee_heights), len(lifetimes)), np.nan)
    su_perigee_matrix = np.full((len(apogee_heights), len(lifetimes)), np.nan)
    
    for i, apogee in enumerate(apogee_heights):
        for j, lifetime in enumerate(lifetimes):
            # S satellites
            perigee_s = fixed_get_disposal_orbits(2024, apogee, "S", pmd_lifetime=lifetime)
            s_perigee_matrix[i, j] = perigee_s[0]
            
            # Su satellites
            perigee_su = fixed_get_disposal_orbits(2024, apogee, "Su", pmd_lifetime=lifetime)
            su_perigee_matrix[i, j] = perigee_su[0]
    
    # Create heatmap plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # S satellites heatmap
    im1 = ax1.imshow(s_perigee_matrix, cmap='viridis', aspect='auto', origin='lower')
    ax1.set_title('S Satellites - Perigee Height Heatmap', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Lifetime (years)', fontsize=12)
    ax1.set_ylabel('Apogee Height (km)', fontsize=12)
    ax1.set_xticks(range(0, len(lifetimes), 4))
    ax1.set_xticklabels(lifetimes[::4])
    ax1.set_yticks(range(len(apogee_heights)))
    ax1.set_yticklabels(apogee_heights)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Perigee Height (km)', fontsize=12)
    
    # Su satellites heatmap
    im2 = ax2.imshow(su_perigee_matrix, cmap='viridis', aspect='auto', origin='lower')
    ax2.set_title('Su Satellites - Perigee Height Heatmap', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Lifetime (years)', fontsize=12)
    ax2.set_ylabel('Apogee Height (km)', fontsize=12)
    ax2.set_xticks(range(0, len(lifetimes), 4))
    ax2.set_xticklabels(lifetimes[::4])
    ax2.set_yticks(range(len(apogee_heights)))
    ax2.set_yticklabels(apogee_heights)
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Perigee Height (km)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_altitude_heatmap_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def test_fixed_function():
    """Test the fixed function with various inputs"""
    
    print("Testing fixed get_disposal_orbits function...")
    
    # Test S satellites
    print("\n--- S Satellites ---")
    for apogee in [500, 1000, 1500]:
        for lifetime in [1, 5, 10, 25]:
            perigee = fixed_get_disposal_orbits(2024, apogee, "S", pmd_lifetime=lifetime)
            print(f"  Apogee {apogee}km, Lifetime {lifetime}yr -> Perigee {perigee[0]:.1f}km")
    
    # Test Su satellites
    print("\n--- Su Satellites ---")
    for apogee in [500, 1000, 1500]:
        for lifetime in [1, 5, 10, 25]:
            perigee = fixed_get_disposal_orbits(2024, apogee, "Su", pmd_lifetime=lifetime)
            print(f"  Apogee {apogee}km, Lifetime {lifetime}yr -> Perigee {perigee[0]:.1f}km")

if __name__ == "__main__":
    print("Creating fixed disposal analysis...")
    
    # Test the fixed function
    test_fixed_function()
    
    # Create line plots
    print("\nCreating line plots...")
    fig1 = create_disposal_altitude_plot_fixed()
    
    # Create heatmap plots
    print("Creating heatmap plots...")
    fig2 = create_heatmap_plot_fixed()
    
    print("Fixed plots saved!")
    print("Files created:")
    print("- disposal_altitude_analysis_fixed.png")
    print("- disposal_altitude_heatmap_fixed.png")
