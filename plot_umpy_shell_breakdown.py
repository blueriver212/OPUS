import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Directory containing simulation results
results_dir = Path("Results/extensive_1")

# Derelict classes to count
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

# Midpoints in altitude space (20 shells)
altitude_midpoints = np.array([
    230., 290., 350., 410., 470., 530., 590., 650., 710.,
    770., 830., 890., 950., 1010., 1070., 1130., 1190., 1250.,
    1310., 1370.
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
    match = re.search(r'N_([\d.]+)kg', species_name)
    if match:
        return float(match.group(1))
    return 0.0

def calculate_umpy_per_shell(pop_ij_array, life_ij_array, mass_i, t_sim, X):
    """Calculate UMPY contribution from each shell"""
    n_shells = len(pop_ij_array)
    e_X = np.exp(X)
    denominator = e_X - 1.0
    
    shell_contributions = np.zeros(n_shells)
    
    for shell_idx in range(n_shells):
        pop_ij = pop_ij_array[shell_idx]
        life_ij = life_ij_array[shell_idx]
        
        exponent = X * (life_ij / t_sim)
        exp_term = np.exp(exponent)
        umpy_factor = (exp_term - 1.0) / denominator
        
        shell_contribution = (mass_i * pop_ij * umpy_factor) / t_sim
        shell_contributions[shell_idx] = shell_contribution
    
    return shell_contributions

def get_final_year_shell_data(species_data, derelict_class):
    """Get the 20 shell values for a specific derelict class for the final year"""
    if derelict_class not in species_data:
        return np.zeros(20)
    
    if FINAL_YEAR not in species_data[derelict_class]:
        return np.zeros(20)
    
    values = species_data[derelict_class][FINAL_YEAR]
    if isinstance(values, list):
        if len(values) == 20:
            return np.array(values)
        elif len(values) < 20:
            return np.pad(values, (0, 20 - len(values)), 'constant')
        else:
            return np.array(values[:20])
    return np.zeros(20)

# Collect data: {(bond, sub): [umpy_contributions_per_shell]}
data_dict = defaultdict(dict)  # {bond: {sub: [contributions]}}

for subfolder in sorted(results_dir.iterdir()):
    if not subfolder.is_dir():
        continue
    
    bond_value, sub_value = extract_bond_sub_values(subfolder.name)
    if bond_value is None or sub_value is None:
        continue
    
    # Skip $0k bond
    if bond_value == 0:
        continue
    
    species_data_files = list(subfolder.glob("species_data_*.json"))
    if not species_data_files:
        continue
    
    try:
        with open(species_data_files[0], 'r') as f:
            species_data = json.load(f)
        
        # Calculate UMPY contribution per shell across all derelict classes
        total_shell_contributions = np.zeros(20)
        
        for derelict_class in derelict_classes:
            pop_ij_array = get_final_year_shell_data(species_data, derelict_class)
            mass_i = extract_mass_from_species_name(derelict_class)
            shell_contributions = calculate_umpy_per_shell(
                pop_ij_array, orbital_lifetimes, mass_i, t_sim, X)
            total_shell_contributions += shell_contributions
        
        data_dict[bond_value][sub_value] = total_shell_contributions
    
    except Exception as e:
        print(f"Error processing {subfolder.name}: {e}")
        continue

# Get unique bond and sub values
bond_values = sorted(set(data_dict.keys()), reverse=True)  # Descending: 2M at top
sub_values = sorted(set(sub for bond_data in data_dict.values() for sub in bond_data.keys()))

print(f"Found {len(bond_values)} bond values: {bond_values}")
print(f"Found {len(sub_values)} sub values: {sub_values}")

# Format bond values for display
def format_bond_value(bond_val):
    """Format bond value for display"""
    if bond_val < 1000000:
        thousands = bond_val // 1000
        return f"${thousands}k"
    else:
        millions = bond_val / 1000000
        return f"${millions:.1f}M"

# Side-by-side layout: 20% column | 60% column; each panel shows all sub rates
sub_compare = [0.2, 0.6]
n_bonds = len(bond_values)
n_cols = 2
n_rows = n_bonds

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
if n_rows == 1:
    axes = axes.reshape(1, -1)

# Color map for all substitution rates (0.0 .. 0.9)
colors_all = plt.cm.tab10(np.linspace(0, 1, len(sub_values)))
col_titles = ['20% Substitution', '60% Substitution']

for col_idx, highlight_sub in enumerate(sub_compare):
    for row_idx, bond_val in enumerate(bond_values):
        ax = axes[row_idx, col_idx]
        # Plot every sub rate in this panel
        for sub_idx, sub_val in enumerate(sub_values):
            if sub_val not in data_dict[bond_val]:
                continue
            shell_contributions = data_dict[bond_val][sub_val]
            lw = 2.5 if sub_val == highlight_sub else 1.2
            alpha = 1.0 if sub_val == highlight_sub else 0.5
            ms = 5 if sub_val == highlight_sub else 2
            ax.plot(altitude_midpoints, shell_contributions,
                    marker='o', linewidth=lw, markersize=ms,
                    label=f'Sub {sub_val:.1f}', color=colors_all[sub_idx], alpha=alpha)
        ax.set_xlabel('Altitude (km)', fontsize=10)
        ax.set_ylabel('UMPY Contribution', fontsize=10)
        bond_label = format_bond_value(bond_val)
        ax.set_title(f'{bond_label} Bond', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8, ncol=2)

# Column headers
for col_idx, title in enumerate(col_titles):
    axes[0, col_idx].set_title(title, fontsize=12, fontweight='bold')

plt.suptitle('UMPY Contribution by Shell: 20% vs 60% (all sub rates shown)', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save plot
output_path = results_dir / "umpy_shell_breakdown.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

plt.show()
