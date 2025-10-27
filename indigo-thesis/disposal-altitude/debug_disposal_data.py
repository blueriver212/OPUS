import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sys
sys.path.append('/Users/indigobrownhall/Code/OPUS/OPUS/utils')
from PostMissionDisposal import get_disposal_orbits

def debug_mat_files():
    """Debug the structure of the .mat files"""
    
    # Load S data
    print("=== S Satellite Data ===")
    s_data = sio.loadmat('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_S.mat', squeeze_me=True, struct_as_record=False)
    lookup_s = s_data['lookup']
    
    print(f"Years: {lookup_s.years}")
    print(f"Apogee alts: {lookup_s.apogee_alts_km}")
    print(f"Perigee alts: {lookup_s.perigee_alts_km}")
    print(f"Lifetimes shape: {lookup_s.lifetimes_years.shape}")
    print(f"Decay alt: {lookup_s.decay_alt_km}")
    
    # Check some sample lifetime data
    print(f"Sample lifetimes for year 2024, apogee 500km: {lookup_s.lifetimes_years[0, 1, :]}")
    print(f"Sample lifetimes for year 2024, apogee 1000km: {lookup_s.lifetimes_years[0, 6, :]}")
    
    # Load Su data
    print("\n=== Su Satellite Data ===")
    su_data = sio.loadmat('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_Su.mat', squeeze_me=True, struct_as_record=False)
    lookup_su = su_data['lookup']
    
    print(f"Years: {lookup_su.years}")
    print(f"Apogee alts: {lookup_su.apogee_alts_km}")
    print(f"Perigee alts: {lookup_su.perigee_alts_km}")
    print(f"Lifetimes shape: {lookup_su.lifetimes_years.shape}")
    print(f"Decay alt: {lookup_su.decay_alt_km}")
    
    # Check some sample lifetime data
    print(f"Sample lifetimes for year 2024, apogee 500km: {lookup_su.lifetimes_years[0, 1, :]}")
    print(f"Sample lifetimes for year 2024, apogee 1000km: {lookup_su.lifetimes_years[0, 6, :]}")

def test_disposal_orbits():
    """Test the get_disposal_orbits function with different parameters"""
    
    print("\n=== Testing get_disposal_orbits function ===")
    
    # Test with S data
    print("Testing S satellites:")
    for apogee in [500, 1000, 1500]:
        for lifetime in [1, 5, 10, 25]:
            try:
                perigee = get_disposal_orbits(2024, apogee, pmd_lifetime=lifetime, 
                                            lookup_path='/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_S.mat')
                print(f"  Apogee {apogee}km, Lifetime {lifetime}yr -> Perigee {perigee[0]:.1f}km")
            except Exception as e:
                print(f"  Apogee {apogee}km, Lifetime {lifetime}yr -> Error: {e}")
    
    # Test with Su data
    print("\nTesting Su satellites:")
    for apogee in [500, 1000, 1500]:
        for lifetime in [1, 5, 10, 25]:
            try:
                perigee = get_disposal_orbits(2024, apogee, pmd_lifetime=lifetime, 
                                            lookup_path='/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_Su.mat')
                print(f"  Apogee {apogee}km, Lifetime {lifetime}yr -> Perigee {perigee[0]:.1f}km")
            except Exception as e:
                print(f"  Apogee {apogee}km, Lifetime {lifetime}yr -> Error: {e}")

def create_simple_plot():
    """Create a simple plot using the raw data from the .mat files"""
    
    # Load data
    s_data = sio.loadmat('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_S.mat', squeeze_me=True, struct_as_record=False)
    lookup_s = s_data['lookup']
    
    su_data = sio.loadmat('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_Su.mat', squeeze_me=True, struct_as_record=False)
    lookup_su = su_data['lookup']
    
    # Get data for year 2024 (index 0)
    s_lifetimes = lookup_s.lifetimes_years[0, :, :]  # [apogee, perigee]
    su_lifetimes = lookup_su.lifetimes_years[0, :, :]
    
    apogee_alts = lookup_s.apogee_alts_km
    perigee_alts = lookup_s.perigee_alts_km
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot S data
    for i, apogee in enumerate(apogee_alts):
        valid_perigees = []
        valid_lifetimes = []
        for j, perigee in enumerate(perigee_alts):
            if not np.isnan(s_lifetimes[i, j]) and s_lifetimes[i, j] > 0:
                valid_perigees.append(perigee)
                valid_lifetimes.append(s_lifetimes[i, j])
        
        if valid_perigees:
            ax1.plot(valid_lifetimes, valid_perigees, 'o-', label=f'Apogee {apogee}km', markersize=4)
    
    ax1.set_title('S Satellites - Perigee vs Lifetime (Raw Data)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Lifetime (years)', fontsize=12)
    ax1.set_ylabel('Perigee Height (km)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot Su data
    for i, apogee in enumerate(apogee_alts):
        valid_perigees = []
        valid_lifetimes = []
        for j, perigee in enumerate(perigee_alts):
            if not np.isnan(su_lifetimes[i, j]) and su_lifetimes[i, j] > 0:
                valid_perigees.append(perigee)
                valid_lifetimes.append(su_lifetimes[i, j])
        
        if valid_perigees:
            ax2.plot(valid_lifetimes, valid_perigees, 'o-', label=f'Apogee {apogee}km', markersize=4)
    
    ax2.set_title('Su Satellites - Perigee vs Lifetime (Raw Data)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Lifetime (years)', fontsize=12)
    ax2.set_ylabel('Perigee Height (km)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('/Users/indigobrownhall/Code/OPUS/disposal_altitude_raw_data.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

if __name__ == "__main__":
    debug_mat_files()
    test_disposal_orbits()
    create_simple_plot()
