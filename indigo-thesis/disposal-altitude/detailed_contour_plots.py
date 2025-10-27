import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib.colors import LogNorm
import matplotlib.patches as patches

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

def create_detailed_contour_plots():
    """Create detailed contour plots for S and Su satellites"""
    
    # Load data
    s_data = load_disposal_lookup('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_S.mat')
    su_data = load_disposal_lookup('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_Su.mat')
    
    # Get data for year 2024 (index 0)
    s_lifetimes = s_data['lifetimes_years'][0, :, :]  # [apogee, perigee]
    su_lifetimes = su_data['lifetimes_years'][0, :, :]
    
    apogee_alts = s_data['apogee_alts_km']
    perigee_alts = s_data['perigee_alts_km']
    
    # Create meshgrids
    perigee_mesh, apogee_mesh = np.meshgrid(perigee_alts, apogee_alts)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Define contour levels
    levels = np.logspace(-1, 2, 20)  # From 0.1 to 100 years
    
    # Plot S satellites
    im1 = ax1.contourf(perigee_mesh, apogee_mesh, s_lifetimes, 
                      levels=levels, cmap='viridis', norm=LogNorm())
    
    # Add contour lines
    cs1 = ax1.contour(perigee_mesh, apogee_mesh, s_lifetimes, 
                     levels=levels, colors='white', alpha=0.6, linewidths=0.5)
    ax1.clabel(cs1, inline=True, fontsize=8, fmt='%.1f')
    
    ax1.set_xlabel('Perigee Height (km)', fontsize=12)
    ax1.set_ylabel('Apogee Height (km)', fontsize=12)
    ax1.set_title('S Satellites - Disposal Lifetime Contours\n(Year 2024)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Disposal Lifetime (years)', fontsize=12)
    
    # Add diagonal line (perigee = apogee)
    max_alt = max(apogee_alts.max(), perigee_alts.max())
    min_alt = min(apogee_alts.min(), perigee_alts.min())
    ax1.plot([min_alt, max_alt], [min_alt, max_alt], 'r--', alpha=0.7, linewidth=2, label='Perigee = Apogee')
    ax1.legend()
    
    # Plot Su satellites
    im2 = ax2.contourf(perigee_mesh, apogee_mesh, su_lifetimes, 
                      levels=levels, cmap='viridis', norm=LogNorm())
    
    # Add contour lines
    cs2 = ax2.contour(perigee_mesh, apogee_mesh, su_lifetimes, 
                     levels=levels, colors='white', alpha=0.6, linewidths=0.5)
    ax2.clabel(cs2, inline=True, fontsize=8, fmt='%.1f')
    
    ax2.set_xlabel('Perigee Height (km)', fontsize=12)
    ax2.set_ylabel('Apogee Height (km)', fontsize=12)
    ax2.set_title('Su Satellites - Disposal Lifetime Contours\n(Year 2024)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Disposal Lifetime (years)', fontsize=12)
    
    # Add diagonal line (perigee = apogee)
    ax2.plot([min_alt, max_alt], [min_alt, max_alt], 'r--', alpha=0.7, linewidth=2, label='Perigee = Apogee')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('/Users/indigobrownhall/Code/OPUS/detailed_contour_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_individual_contour_plots():
    """Create individual detailed contour plots for each satellite type"""
    
    # Load data
    s_data = load_disposal_lookup('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_S.mat')
    su_data = load_disposal_lookup('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_Su.mat')
    
    # Get data for year 2024
    s_lifetimes = s_data['lifetimes_years'][0, :, :]
    su_lifetimes = su_data['lifetimes_years'][0, :, :]
    
    apogee_alts = s_data['apogee_alts_km']
    perigee_alts = s_data['perigee_alts_km']
    
    # Create meshgrids
    perigee_mesh, apogee_mesh = np.meshgrid(perigee_alts, apogee_alts)
    
    # Define contour levels
    levels = np.logspace(-1, 2, 20)
    
    # S Satellites detailed plot
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 10))
    
    im1 = ax1.contourf(perigee_mesh, apogee_mesh, s_lifetimes, 
                      levels=levels, cmap='plasma', norm=LogNorm())
    
    # Add contour lines with labels
    cs1 = ax1.contour(perigee_mesh, apogee_mesh, s_lifetimes, 
                     levels=levels, colors='white', alpha=0.8, linewidths=1)
    ax1.clabel(cs1, inline=True, fontsize=10, fmt='%.1f')
    
    # Add specific lifetime contours
    specific_levels = [0.1, 0.5, 1, 2, 5, 10, 25, 50]
    cs_specific = ax1.contour(perigee_mesh, apogee_mesh, s_lifetimes, 
                             levels=specific_levels, colors='red', alpha=0.9, linewidths=2)
    ax1.clabel(cs_specific, inline=True, fontsize=12, fmt='%.1f', colors='red')
    
    ax1.set_xlabel('Perigee Height (km)', fontsize=14)
    ax1.set_ylabel('Apogee Height (km)', fontsize=14)
    ax1.set_title('S Satellites - Disposal Lifetime Contours\n(Year 2024)', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Disposal Lifetime (years)', fontsize=14)
    
    # Add diagonal line and annotations
    max_alt = max(apogee_alts.max(), perigee_alts.max())
    min_alt = min(apogee_alts.min(), perigee_alts.min())
    ax1.plot([min_alt, max_alt], [min_alt, max_alt], 'k--', alpha=0.7, linewidth=2, label='Perigee = Apogee')
    ax1.legend(fontsize=12)
    
    # Add text annotations
    ax1.text(0.02, 0.98, 'Lower perigee = Shorter lifetime', transform=ax1.transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/Users/indigobrownhall/Code/OPUS/S_satellites_contour_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Su Satellites detailed plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))
    
    im2 = ax2.contourf(perigee_mesh, apogee_mesh, su_lifetimes, 
                      levels=levels, cmap='plasma', norm=LogNorm())
    
    # Add contour lines with labels
    cs2 = ax2.contour(perigee_mesh, apogee_mesh, su_lifetimes, 
                     levels=levels, colors='white', alpha=0.8, linewidths=1)
    ax2.clabel(cs2, inline=True, fontsize=10, fmt='%.1f')
    
    # Add specific lifetime contours
    cs_specific2 = ax2.contour(perigee_mesh, apogee_mesh, su_lifetimes, 
                              levels=specific_levels, colors='red', alpha=0.9, linewidths=2)
    ax2.clabel(cs_specific2, inline=True, fontsize=12, fmt='%.1f', colors='red')
    
    ax2.set_xlabel('Perigee Height (km)', fontsize=14)
    ax2.set_ylabel('Apogee Height (km)', fontsize=14)
    ax2.set_title('Su Satellites - Disposal Lifetime Contours\n(Year 2024)', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Disposal Lifetime (years)', fontsize=14)
    
    # Add diagonal line and annotations
    ax2.plot([min_alt, max_alt], [min_alt, max_alt], 'k--', alpha=0.7, linewidth=2, label='Perigee = Apogee')
    ax2.legend(fontsize=12)
    
    # Add text annotations
    ax2.text(0.02, 0.98, 'Lower perigee = Shorter lifetime', transform=ax2.transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/Users/indigobrownhall/Code/OPUS/Su_satellites_contour_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig1, fig2

def create_comparison_heatmap():
    """Create a side-by-side comparison heatmap"""
    
    # Load data
    s_data = load_disposal_lookup('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_S.mat')
    su_data = load_disposal_lookup('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_Su.mat')
    
    # Get data for year 2024
    s_lifetimes = s_data['lifetimes_years'][0, :, :]
    su_lifetimes = su_data['lifetimes_years'][0, :, :]
    
    apogee_alts = s_data['apogee_alts_km']
    perigee_alts = s_data['perigee_alts_km']
    
    # Create meshgrids
    perigee_mesh, apogee_mesh = np.meshgrid(perigee_alts, apogee_alts)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Define levels for consistent colorbar
    all_lifetimes = np.concatenate([s_lifetimes.flatten(), su_lifetimes.flatten()])
    valid_lifetimes = all_lifetimes[~np.isnan(all_lifetimes)]
    vmin, vmax = np.percentile(valid_lifetimes, [5, 95])
    
    # S Satellites heatmap
    im1 = ax1.imshow(s_lifetimes, cmap='plasma', norm=LogNorm(vmin=vmin, vmax=vmax), 
                    origin='lower', aspect='auto')
    ax1.set_xlabel('Perigee Height (km)', fontsize=14)
    ax1.set_ylabel('Apogee Height (km)', fontsize=14)
    ax1.set_title('S Satellites - Disposal Lifetime Heatmap\n(Year 2024)', fontsize=16, fontweight='bold')
    
    # Set ticks
    ax1.set_xticks(range(len(perigee_alts)))
    ax1.set_xticklabels(perigee_alts)
    ax1.set_yticks(range(len(apogee_alts)))
    ax1.set_yticklabels(apogee_alts)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Disposal Lifetime (years)', fontsize=14)
    
    # Su Satellites heatmap
    im2 = ax2.imshow(su_lifetimes, cmap='plasma', norm=LogNorm(vmin=vmin, vmax=vmax), 
                    origin='lower', aspect='auto')
    ax2.set_xlabel('Perigee Height (km)', fontsize=14)
    ax2.set_ylabel('Apogee Height (km)', fontsize=14)
    ax2.set_title('Su Satellites - Disposal Lifetime Heatmap\n(Year 2024)', fontsize=16, fontweight='bold')
    
    # Set ticks
    ax2.set_xticks(range(len(perigee_alts)))
    ax2.set_xticklabels(perigee_alts)
    ax2.set_yticks(range(len(apogee_alts)))
    ax2.set_yticklabels(apogee_alts)
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Disposal Lifetime (years)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('/Users/indigobrownhall/Code/OPUS/satellites_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_lifetime_vs_altitude_analysis():
    """Create analysis showing how lifetime varies with altitude for different perigee heights"""
    
    # Load data
    s_data = load_disposal_lookup('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_S.mat')
    su_data = load_disposal_lookup('/Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude/disposal_lookup_Su.mat')
    
    # Get data for year 2024
    s_lifetimes = s_data['lifetimes_years'][0, :, :]
    su_lifetimes = su_data['lifetimes_years'][0, :, :]
    
    apogee_alts = s_data['apogee_alts_km']
    perigee_alts = s_data['perigee_alts_km']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot S satellites
    for i, perigee in enumerate(perigee_alts):
        lifetimes = s_lifetimes[:, i]
        valid_mask = ~np.isnan(lifetimes)
        if np.any(valid_mask):
            ax1.plot(apogee_alts[valid_mask], lifetimes[valid_mask], 'o-', 
                    label=f'Perigee {perigee} km', markersize=6, linewidth=2)
    
    ax1.set_xlabel('Apogee Height (km)', fontsize=12)
    ax1.set_ylabel('Disposal Lifetime (years)', fontsize=12)
    ax1.set_title('S Satellites - Lifetime vs Apogee Height\n(Year 2024)', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot Su satellites
    for i, perigee in enumerate(perigee_alts):
        lifetimes = su_lifetimes[:, i]
        valid_mask = ~np.isnan(lifetimes)
        if np.any(valid_mask):
            ax2.plot(apogee_alts[valid_mask], lifetimes[valid_mask], 'o-', 
                    label=f'Perigee {perigee} km', markersize=6, linewidth=2)
    
    ax2.set_xlabel('Apogee Height (km)', fontsize=12)
    ax2.set_ylabel('Disposal Lifetime (years)', fontsize=12)
    ax2.set_title('Su Satellites - Lifetime vs Apogee Height\n(Year 2024)', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('/Users/indigobrownhall/Code/OPUS/lifetime_vs_altitude_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

if __name__ == "__main__":
    print("Creating detailed contour plots...")
    
    # Create side-by-side contour plots
    print("Creating side-by-side contour plots...")
    fig1 = create_detailed_contour_plots()
    
    # Create individual detailed plots
    print("Creating individual detailed plots...")
    fig2, fig3 = create_individual_contour_plots()
    
    # Create comparison heatmap
    print("Creating comparison heatmap...")
    fig4 = create_comparison_heatmap()
    
    # Create lifetime vs altitude analysis
    print("Creating lifetime vs altitude analysis...")
    fig5 = create_lifetime_vs_altitude_analysis()
    
    print("All plots saved!")
    print("Files created:")
    print("- detailed_contour_plots.png")
    print("- S_satellites_contour_detailed.png")
    print("- Su_satellites_contour_detailed.png")
    print("- satellites_comparison_heatmap.png")
    print("- lifetime_vs_altitude_analysis.png")
