import json
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from pathlib import Path
from collections import defaultdict

# Directory containing simulation results
results_dir = Path("Results/extensive_1")

# Derelict classes (exclude these from "other debris")
derelict_classes = [
    "N_521kg",
    "N_521.01kg",
    "N_521.02kg",
    "N_700kg",
    "N_700.01kg",
    "N_700.02kg"
]

# Final year to use
FINAL_YEAR = "2049"

# UMPY calculation parameters
t_sim = 100.0
X = 4.0

# Orbital lifetimes for each shell (20 shells)
orbital_lifetimes = np.array([
    0.00020572928539568173, 0.0011872686787319695, 0.004601778372329309, 
    0.01504391781075165, 0.045015820859178726, 0.12555642200397685, 
    0.3308689573678134, 0.8362918548487709, 1.9965228046698078, 
    4.392851313568226, 9.087513579506581, 16.65340806861344, 
    28.25238479409053, 44.34279111315705, 65.1208810339771, 
    91.00816248631243, 123.2619892340123, 163.44937999056947, 
    213.52333695907197, 275.91798563301364
])

def extract_bond_sub_values(folder_name):
    """Extract Bond and Sub values from folder name"""
    match = re.search(r'Bond_([\d]+)_Sub_([\d.]+)', folder_name)
    if match:
        bond_value = int(match.group(1))
        sub_value = float(match.group(2))
        return bond_value, sub_value
    return None, None

def extract_mass_from_species_name(species_name):
    """Extract mass from species name"""
    if species_name == "B":
        return 500.0
    match = re.search(r'N_([\d.]+)kg', species_name)
    if match:
        return float(match.group(1))
    return 0.0

def calculate_umpy(pop_ij_array, life_ij_array, mass_i, t_sim, X):
    """Calculate UMPY for a species"""
    n_shells = len(pop_ij_array)
    e_X = np.exp(X)
    denominator = e_X - 1.0
    total_sum = 0.0
    for shell_idx in range(n_shells):
        pop_ij = pop_ij_array[shell_idx]
        life_ij = life_ij_array[shell_idx]
        exponent = X * (life_ij / t_sim)
        exp_term = np.exp(exponent)
        umpy_factor = (exp_term - 1.0) / denominator
        shell_contribution = (mass_i * pop_ij * umpy_factor) / t_sim
        total_sum += shell_contribution
    return total_sum

def get_final_year_shell_data(species_data, species_name):
    """Get the 20 shell values for a specific species for the final year"""
    if species_name not in species_data:
        return np.zeros(20)
    if FINAL_YEAR not in species_data[species_name]:
        return np.zeros(20)
    values = species_data[species_name][FINAL_YEAR]
    if isinstance(values, list):
        if len(values) == 20:
            return np.array(values)
        elif len(values) < 20:
            return np.pad(values, (0, 20 - len(values)), 'constant')
        else:
            return np.array(values[:20])
    return np.zeros(20)

def get_other_debris_species(species_data):
    """Get species that are non-S and NOT derelicts (other debris)"""
    non_s = [name for name in species_data.keys() if not name.startswith('S')]
    other_debris = [name for name in non_s if name not in derelict_classes]
    return other_debris

# Collect data: sum UMPY for other debris only
data_dict = {}  # {(bond, sub): umpy_value}
baseline_umpy = None

for subfolder in sorted(results_dir.iterdir()):
    if not subfolder.is_dir():
        continue
    bond_value, sub_value = extract_bond_sub_values(subfolder.name)
    if bond_value is None or sub_value is None:
        continue
    species_data_files = list(subfolder.glob("species_data_*.json"))
    if not species_data_files:
        continue
    try:
        with open(species_data_files[0], 'r') as f:
            species_data = json.load(f)
        other_debris_species = get_other_debris_species(species_data)
        if len(other_debris_species) == 0:
            print(f"Warning: No other-debris species in {subfolder.name}")
            continue
        total_umpy = 0.0
        for species_name in other_debris_species:
            pop_ij_array = get_final_year_shell_data(species_data, species_name)
            mass_i = extract_mass_from_species_name(species_name)
            umpy = calculate_umpy(pop_ij_array, orbital_lifetimes, mass_i, t_sim, X)
            total_umpy += umpy
        data_dict[(bond_value, sub_value)] = total_umpy
        if bond_value == 0 and sub_value == 0.0:
            baseline_umpy = total_umpy
            print(f"Baseline (Bond_0_Sub_0) other-debris UMPY = {baseline_umpy:.6f}")
            print(f"  Other-debris species: {other_debris_species}")
    except Exception as e:
        print(f"Error processing {species_data_files[0]}: {e}")
        continue

if baseline_umpy is None:
    print("ERROR: Could not find baseline (Bond_0_Sub_0)")
    exit(1)

bond_values = sorted(set(b for b, s in data_dict.keys()), reverse=True)
sub_values = sorted(set(s for b, s in data_dict.keys()))

# Raw UMPY matrix
matrix_raw = np.full((len(bond_values), len(sub_values)), np.nan)
for i, bond_val in enumerate(bond_values):
    for j, sub_val in enumerate(sub_values):
        if (bond_val, sub_val) in data_dict:
            matrix_raw[i, j] = data_dict[(bond_val, sub_val)]

# % change matrix
matrix_pct = np.full((len(bond_values), len(sub_values)), np.nan)
for i, bond_val in enumerate(bond_values):
    for j, sub_val in enumerate(sub_values):
        if (bond_val, sub_val) in data_dict:
            umpy_val = data_dict[(bond_val, sub_val)]
            matrix_pct[i, j] = ((umpy_val - baseline_umpy) / baseline_umpy) * 100

def format_bond_value(bond_val):
    if bond_val == 0:
        return "$0k"
    elif bond_val < 1000000:
        return f"${bond_val // 1000}k"
    else:
        m, r = bond_val // 1000000, (bond_val % 1000000) // 100000
        return f"${m}.{r}M" if r else f"${m}.0M"

def plot_heatmap(ax, matrix, title, cbar_label, vmin=None, vmax=None, discrete_levels=10):
    if vmin is None:
        vmin = np.nanmin(matrix)
    if vmax is None:
        vmax = np.nanmax(matrix)
    levels = np.linspace(vmin, vmax, discrete_levels + 1)
    cmap = plt.cm.RdYlGn_r
    norm = BoundaryNorm(levels, cmap.N)
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_xticks(range(len(sub_values)))
    ax.set_xticklabels([f'{sv:.1f}' for sv in sub_values])
    ax.set_yticks(range(len(bond_values)))
    ax.set_yticklabels([format_bond_value(bv) for bv in bond_values])
    ax.set_xlabel('Substitution Rate', fontsize=12)
    ax.set_ylabel('Bond Amount', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, boundaries=levels, ticks=levels[:-1] + np.diff(levels)/2)
    cbar.set_label(cbar_label, fontsize=10, rotation=270, labelpad=15)
    cbar.ax.set_yticklabels([f'{t:.1f}' for t in levels[:-1] + np.diff(levels)/2])
    vcenter = (vmin + vmax) / 2
    for i in range(len(bond_values)):
        for j in range(len(sub_values)):
            val = matrix[i, j]
            if not np.isnan(val):
                txt = f'{val:.1f}'
                nv = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                clr = 'white' if nv < 0.3 or nv > 0.7 else 'black'
                ax.text(j, i, txt, ha='center', va='center', color=clr, fontsize=9, fontweight='bold')

fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# Plot 1: Raw UMPY
plot_heatmap(axes[0], matrix_raw, 'Other Debris UMPY (Final Year)', 'UMPY Value')

# Plot 2: % change
plot_heatmap(axes[1], matrix_pct, 'Other Debris UMPY % Change (vs Bond $0k, Sub 0.0)', '% Change')

plt.suptitle('Other Debris (Non-Derelict) UMPY', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

out_raw = results_dir / "umpy_heatmap_other_debris.png"
out_pct = results_dir / "umpy_heatmap_other_debris_percent_change.png"
plt.savefig(out_raw, dpi=300, bbox_inches='tight')
print(f"\nRaw UMPY heatmap saved to: {out_raw}")

# Save second plot separately (percent change only) as requested
fig2, ax2 = plt.subplots(figsize=(12, 10))
plot_heatmap(ax2, matrix_pct, 'Other Debris UMPY % Change (vs Bond $0k, Sub 0.0)', '% Change')
plt.tight_layout()
plt.savefig(out_pct, dpi=300, bbox_inches='tight')
print(f"% change heatmap saved to: {out_pct}")
