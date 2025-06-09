import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

def plot_rmse_heatmap(truth_grids, flux_grids, semi_major_axis_points, eccentricity_points, output_path):
    # Ensure both inputs are 3D lists of 2D arrays: (time, sma, ecc)
    n_snapshots = len(truth_grids)
    if len(truth_grids) != len(flux_grids):
        raise ValueError(f"Mismatched input lengths: {len(truth_grids)} truth grids vs {len(flux_grids)} flux grids.")

    rmse_matrix = np.zeros_like(truth_grids[0], dtype=float)

    for A, F in zip(truth_grids, flux_grids):
        rmse_matrix += (A - F) ** 2

    rmse_matrix = np.sqrt(rmse_matrix / n_snapshots)

    # Set up plot grid
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmin=np.min(rmse_matrix), vmax=np.max(rmse_matrix))

    im = ax.imshow(rmse_matrix.T, origin='lower', aspect='auto', cmap=cmap, norm=norm,
                   extent=[semi_major_axis_points[0]/1e4, semi_major_axis_points[-1]/1e4,
                           eccentricity_points[0], eccentricity_points[-1]])

    ax.set_xlabel('Semi-Major Axis [×10⁴ km]')
    ax.set_ylabel('Eccentricity')
    ax.set_title('RMSE Heatmap – Flux vs Reference')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('RMSE (objects)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved RMSE heatmap to: {output_path}")
    plt.close(fig)