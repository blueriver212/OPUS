import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

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

def find_perigee_for_lifetime(lifetimes_data, perigee_alts, target_lifetime, apogee_idx):
    """
    Find the perigee altitude that gives the closest lifetime to target_lifetime
    for a given apogee altitude.
    """
    lifetimes_for_apogee = lifetimes_data[apogee_idx, :]  # [perigee]
    
    # Find valid (non-NaN) lifetimes
    valid_mask = ~np.isnan(lifetimes_for_apogee)
    if not np.any(valid_mask):
        return np.nan
    
    valid_lifetimes = lifetimes_for_apogee[valid_mask]
    valid_perigees = perigee_alts[valid_mask]
    
    # If target lifetime is less than minimum available, return minimum perigee
    if target_lifetime <= np.min(valid_lifetimes):
        return np.min(valid_perigees)
    
    # If target lifetime is greater than maximum available, return maximum perigee
    if target_lifetime >= np.max(valid_lifetimes):
        return np.max(valid_perigees)
    
    # Interpolate to find perigee for target lifetime
    perigee = np.interp(target_lifetime, valid_lifetimes, valid_perigees)
    return perigee

def create_disposal_altitude_plot():
    """Create plot showing perigee heights for varying lifetimes"""
    
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
            # Find perigee for this lifetime using the raw data
            perigee = find_perigee_for_lifetime(s_data['lifetimes_years'][0, :, :],  # Year 2024
                                              s_data['perigee_alts_km'], 
                                              lifetime, i)
            perigee_heights.append(perigee)
        
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
            # Find perigee for this lifetime using the raw data
            perigee = find_perigee_for_lifetime(su_data['lifetimes_years'][0, :, :],  # Year 2024
                                              su_data['perigee_alts_km'], 
                                              lifetime, i)
            perigee_heights.append(perigee)
        
        # Plot only if we have valid data
        valid_mask = ~np.isnan(perigee_heights)
        if np.any(valid_mask):
            ax2.plot(np.array(lifetimes)[valid_mask], np.array(perigee_heights)[valid_mask], 
                    'o-', color=colors[i], label=f'Apogee {apogee} km', markersize=4)
    
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.set_xlim(1, 25)
    
    plt.tight_layout()
    plt.savefig('/Users/indigobrownhall/Code/OPUS/disposal_altitude_analysis_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_heatmap_plot():
    """Create heatmap showing perigee heights for different apogee/lifetime combinations"""
    
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
            perigee_s = find_perigee_for_lifetime(s_data['lifetimes_years'][0, :, :],  # Year 2024
                                                s_data['perigee_alts_km'], 
                                                lifetime, i)
            s_perigee_matrix[i, j] = perigee_s
            
            # Su satellites
            perigee_su = find_perigee_for_lifetime(su_data['lifetimes_years'][0, :, :],  # Year 2024
                                                 su_data['perigee_alts_km'], 
                                                 lifetime, i)
            su_perigee_matrix[i, j] = perigee_su
    
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
    plt.savefig('/Users/indigobrownhall/Code/OPUS/disposal_altitude_heatmap_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_detailed_analysis():
    """Create a more detailed analysis showing the relationship between apogee, perigee, and lifetime"""
    
    # Load data
    s_data = load_disposal_lookup('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_S.mat')
    su_data = load_disposal_lookup('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_Su.mat')
    
    # Create 3D surface plots
    fig = plt.figure(figsize=(20, 6))
    
    # S satellites - 3D surface
    ax1 = fig.add_subplot(131, projection='3d')
    
    apogee_mesh, perigee_mesh = np.meshgrid(s_data['apogee_alts_km'], s_data['perigee_alts_km'], indexing='ij')
    lifetime_mesh = s_data['lifetimes_years'][0, :, :]  # Year 2024
    
    # Mask out NaN values
    valid_mask = ~np.isnan(lifetime_mesh)
    
    surf1 = ax1.plot_surface(apogee_mesh, perigee_mesh, lifetime_mesh, 
                           cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
    ax1.set_xlabel('Apogee Height (km)')
    ax1.set_ylabel('Perigee Height (km)')
    ax1.set_zlabel('Lifetime (years)')
    ax1.set_title('S Satellites - Lifetime Surface')
    
    # Su satellites - 3D surface
    ax2 = fig.add_subplot(132, projection='3d')
    
    lifetime_mesh_su = su_data['lifetimes_years'][0, :, :]  # Year 2024
    
    surf2 = ax2.plot_surface(apogee_mesh, perigee_mesh, lifetime_mesh_su, 
                           cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
    ax2.set_xlabel('Apogee Height (km)')
    ax2.set_ylabel('Perigee Height (km)')
    ax2.set_zlabel('Lifetime (years)')
    ax2.set_title('Su Satellites - Lifetime Surface')
    
    # Contour plot comparison
    ax3 = fig.add_subplot(133)
    
    # Create contour plot for S satellites
    levels = np.arange(0, 50, 5)
    cs1 = ax3.contour(apogee_mesh, perigee_mesh, lifetime_mesh, levels=levels, colors='blue', alpha=0.7)
    ax3.clabel(cs1, inline=True, fontsize=8, fmt='%.0f')
    
    # Create contour plot for Su satellites
    cs2 = ax3.contour(apogee_mesh, perigee_mesh, lifetime_mesh_su, levels=levels, colors='red', alpha=0.7, linestyles='--')
    ax3.clabel(cs2, inline=True, fontsize=8, fmt='%.0f')
    
    ax3.set_xlabel('Apogee Height (km)')
    ax3.set_ylabel('Perigee Height (km)')
    ax3.set_title('Lifetime Contours (S=blue, Su=red dashed)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/indigobrownhall/Code/OPUS/disposal_altitude_3d_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

if __name__ == "__main__":
    print("Creating disposal altitude analysis plots...")
    
    # Create line plots
    print("Creating line plots...")
    fig1 = create_disposal_altitude_plot()
    
    # Create heatmap plots
    print("Creating heatmap plots...")
    fig2 = create_heatmap_plot()
    
    # Create detailed 3D analysis
    print("Creating 3D analysis...")
    fig3 = create_detailed_analysis()
    
    print("Plots saved as:")
    print("- disposal_altitude_analysis_fixed.png")
    print("- disposal_altitude_heatmap_fixed.png")
    print("- disposal_altitude_3d_analysis.png")
