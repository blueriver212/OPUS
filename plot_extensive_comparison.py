"""
4x2 comparison: 20% (extensive_2) vs 60% (extensive_1) market with bonds.
Rows: 1) % change derelict count, 2) % change derelict UMPY,
      3) % change UMPY all non-S, 4) % change UMPY other debris (non-derelict).
Columns: left = 20%, right = 60%.
"""
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from collections import defaultdict

FINAL_YEAR = "2049"
t_sim = 100.0
X = 4.0

derelict_classes = [
    "N_521kg", "N_521.01kg", "N_521.02kg",
    "N_700kg", "N_700.01kg", "N_700.02kg"
]

orbital_lifetimes = np.array([
    0.00020572928539568173, 0.0011872686787319695, 0.004601778372329309,
    0.01504391781075165, 0.045015820859178726, 0.12555642200397685,
    0.3308689573678134, 0.8362918548487709, 1.9965228046698078,
    4.392851313568226, 9.087513579506581, 16.65340806861344,
    28.25238479409053, 44.34279111315705, 65.1208810339771,
    91.00816248631243, 123.2619892340123, 163.44937999056947,
    213.52333695907197, 275.91798563301364
])

def extract_bond_sub(folder_name):
    m = re.search(r'Bond_([\d]+)_Sub_([\d.]+)', folder_name)
    return (int(m.group(1)), float(m.group(2))) if m else (None, None)

def mass_from_name(name):
    if name == "B":
        return 500.0
    m = re.search(r'N_([\d.]+)kg', name)
    return float(m.group(1)) if m else 0.0

def umpy_one_species(pop_ij, life_ij, mass_i):
    e_X = np.exp(X)
    denom = e_X - 1.0
    total = 0.0
    for i in range(len(pop_ij)):
        exp_term = np.exp(X * (life_ij[i] / t_sim))
        total += (mass_i * pop_ij[i] * (exp_term - 1.0) / denom) / t_sim
    return total

def get_shell_data(species_data, name):
    if name not in species_data or FINAL_YEAR not in species_data[name]:
        return np.zeros(20)
    v = species_data[name][FINAL_YEAR]
    if not isinstance(v, list):
        return np.zeros(20)
    v = np.array(v[:20]) if len(v) >= 20 else np.pad(v, (0, 20 - len(v)))
    return v

def compute_metrics_for_folder(species_data):
    """Returns (derelict_count, derelict_umpy, all_n_umpy, other_debris_umpy)."""
    derelict_count = 0.0
    derelict_umpy = 0.0
    all_n_umpy = 0.0  # all species beginning with N
    other_debris_umpy = 0.0
    non_s = [n for n in species_data.keys() if not n.startswith('S')]
    n_species = [n for n in species_data.keys() if n.startswith('N')]
    other_debris = [n for n in non_s if n not in derelict_classes]
    for name in derelict_classes:
        shell = get_shell_data(species_data, name)
        derelict_count += np.sum(shell)
        derelict_umpy += umpy_one_species(shell, orbital_lifetimes, mass_from_name(name))
    for name in n_species:
        shell = get_shell_data(species_data, name)
        all_n_umpy += umpy_one_species(shell, orbital_lifetimes, mass_from_name(name))
    for name in other_debris:
        shell = get_shell_data(species_data, name)
        other_debris_umpy += umpy_one_species(shell, orbital_lifetimes, mass_from_name(name))
    return derelict_count, derelict_umpy, all_n_umpy, other_debris_umpy

def load_scenario(results_dir):
    """Returns (bond_values, sub_values, data_dict).
    data_dict[(b,s)] = (derelict_count, derelict_umpy, all_n_umpy, other_debris_umpy).
    """
    data = {}
    for subfolder in sorted(Path(results_dir).iterdir()):
        if not subfolder.is_dir():
            continue
        b, s = extract_bond_sub(subfolder.name)
        if b is None:
            continue
        files = list(subfolder.glob("species_data_*.json"))
        if not files:
            continue
        try:
            with open(files[0], 'r') as f:
                sd = json.load(f)
            data[(b, s)] = compute_metrics_for_folder(sd)
        except Exception as e:
            print(f"Error {subfolder.name}: {e}")
    bonds = sorted(set(k[0] for k in data if k[0] > 0), reverse=True)  # Unique bond values, exclude $0k
    subs = sorted(set(k[1] for k in data))
    return bonds, subs, data

def pct_change_matrix(bonds, subs, data, metric_idx, baseline_key=(0, 0.0)):
    """metric_idx: 0=derelict_count, 1=derelict_umpy, 2=all_n_umpy, 3=other_debris_umpy."""
    base = data.get(baseline_key, (0, 0, 0, 0))[metric_idx]
    if base == 0:
        base = 1e-20
    mat = np.full((len(bonds), len(subs)), np.nan)
    for i, b in enumerate(bonds):
        for j, s in enumerate(subs):
            if (b, s) in data:
                val = data[(b, s)][metric_idx]
                mat[i, j] = ((val - base) / base) * 100
    return mat

def format_bond(b):
    if b == 0:
        return "$0k"
    if b < 1e6:
        return f"${b//1000}k"
    # Millions: one decimal place e.g. 1500000 -> $1.5M
    val_m = b / 1e6
    return f"${val_m:.1f}M"

def plot_heatmap(ax, matrix, bonds, subs, title, vmin=None, vmax=None, cmap_name='RdYlGn_r', white_text_on_dark=False):
    if vmin is None:
        vmin = np.nanmin(matrix)
    if vmax is None:
        vmax = np.nanmax(matrix)
    cmap = plt.colormaps.get_cmap(cmap_name)
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    # One tick per column/row so one label per heatmap row/column (larger text)
    ax.set_xticks(range(len(subs)))
    ax.set_xticklabels([f'{s:.1f}' for s in subs], fontsize=12)
    ax.set_yticks(range(len(bonds)))
    ax.set_yticklabels([format_bond(b) for b in bonds], fontsize=11)
    ax.set_xlabel('Substitution Rate', fontsize=14)
    ax.set_ylabel('Bond Fee', fontsize=14)
    ax.set_title(title, fontsize=14)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('% Change', fontsize=12, rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=11)
    # Limit colorbar to ~6 ticks so it stays readable (continuous scale)
    cbar.locator = MaxNLocator(nbins=6)
    cbar.update_ticks()
    # One value per cell; white text on dark cells when white_text_on_dark=True (e.g. top row)
    for i in range(len(bonds)):
        for j in range(len(subs)):
            v = matrix[i, j]
            if not np.isnan(v):
                nv = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                clr = 'white' if (white_text_on_dark and nv < 0.4) else 'black'  # white on dark blue
                ax.text(j, i, f'{v:.1f}', ha='center', va='center', color=clr, fontsize=10)
    return im

# Paths
base_path = Path("Results")
dir_20 = base_path / "extensive_2"   # 20% market with bonds
dir_60 = base_path / "extensive_1"   # 60% market with bonds

print("Loading 20% (extensive_2)...")
bonds_20, subs_20, data_20 = load_scenario(dir_20)
print("Loading 60% (extensive_1)...")
bonds_60, subs_60, data_60 = load_scenario(dir_60)

# Subplot titles: row 0 = Total Derelicts, row 1 = Derelict UMPY
subplot_titles_20 = ['20% Bonded: Total Derelicts', '20% Bonded: Derelict UMPY']
subplot_titles_60 = ['60% Bonded: Total Derelicts', '60% Bonded: Derelict UMPY']
metric_indices = [0, 1]
# First row: light blue = small change, dark blue = big change (inverted: dark at vmin, light at vmax)
_dark_to_light_blue = LinearSegmentedColormap.from_list('dark_light_blue', ['#00008b', '#add8e6'])  # dark blue -> light blue
if 'dark_light_blue' not in plt.colormaps():
    plt.colormaps.register(_dark_to_light_blue)
colormaps = ['dark_light_blue', 'Greens_r']

# Build matrices for each scenario and metric (first 2 only)
matrices_20 = [pct_change_matrix(bonds_20, subs_20, data_20, k) for k in metric_indices]
matrices_60 = [pct_change_matrix(bonds_60, subs_60, data_60, k) for k in metric_indices]

# Global color scale per row (same scale for left and right in that row)
vmin_per_row = [min(np.nanmin(matrices_20[k]), np.nanmin(matrices_60[k])) for k in range(2)]
vmax_per_row = [max(np.nanmax(matrices_20[k]), np.nanmax(matrices_60[k])) for k in range(2)]
# Second row: fix colorbar max at 100
vmax_per_row[1] = 100

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for row in range(2):
    # Left: 20%; top row uses white text on dark blue cells
    plot_heatmap(axes[row, 0], matrices_20[row], bonds_20, subs_20,
                 subplot_titles_20[row],
                 vmin=vmin_per_row[row], vmax=vmax_per_row[row],
                 cmap_name=colormaps[row], white_text_on_dark=(row == 0))
    # Right: 60%
    plot_heatmap(axes[row, 1], matrices_60[row], bonds_60, subs_60,
                 subplot_titles_60[row],
                 vmin=vmin_per_row[row], vmax=vmax_per_row[row],
                 cmap_name=colormaps[row], white_text_on_dark=(row == 0))

# All y-axes say Bond Fee (set in plot_heatmap); no overall title
plt.tight_layout()

out = base_path / "extensive_comparison_2x2.png"
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f"Saved: {out}")
