import json
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from pathlib import Path
from collections import defaultdict

# Directory containing simulation results
results_dir = Path("Results/extensive_1")

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
    # For B species, use 500kg
    if species_name == "B":
        return 500.0
    
    # For N_*kg species, extract the mass value
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

def get_all_non_s_species(species_data):
    """Get all species names that don't start with 'S'"""
    return [name for name in species_data.keys() if not name.startswith('S')]

# Collect data from all simulations
# Structure: {species_name: {(bond, sub): umpy_value}}
species_data_dict = defaultdict(dict)
baseline_umpy_by_species = {}

# First pass: get all species names from baseline
baseline_folder = None
for subfolder in sorted(results_dir.iterdir()):
    if not subfolder.is_dir():
        continue
    bond_value, sub_value = extract_bond_sub_values(subfolder.name)
    if bond_value == 0 and sub_value == 0.0:
        baseline_folder = subfolder
        break

if baseline_folder is None:
    print("ERROR: Could not find baseline folder (Bond_0_Sub_0)")
    exit(1)

species_data_files = list(baseline_folder.glob("species_data_*.json"))
if not species_data_files:
    print("ERROR: No species_data file found in baseline folder")
    exit(1)

with open(species_data_files[0], 'r') as f:
    baseline_species_data = json.load(f)
    all_species = get_all_non_s_species(baseline_species_data)
    print(f"Found {len(all_species)} non-S species: {all_species}")

# Collect data for all simulations
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
        
        # Calculate UMPY for each species separately
        for species_name in all_species:
            pop_ij_array = get_final_year_shell_data(species_data, species_name)
            mass_i = extract_mass_from_species_name(species_name)
            umpy = calculate_umpy(pop_ij_array, orbital_lifetimes, mass_i, t_sim, X)
            species_data_dict[species_name][(bond_value, sub_value)] = umpy
            
            # Store baseline value (Bond_0_Sub_0)
            if bond_value == 0 and sub_value == 0.0:
                baseline_umpy_by_species[species_name] = umpy
    
    except Exception as e:
        print(f"Error processing {species_data_files[0]}: {e}")
        continue

# Extract unique bond and sub values
bond_values = sorted(set(b for species_dict in species_data_dict.values() 
                         for b, s in species_dict.keys()), reverse=True)
sub_values = sorted(set(s for species_dict in species_data_dict.values() 
                        for b, s in species_dict.keys()))

print(f"\nFound {len(bond_values)} bond values: {bond_values}")
print(f"Found {len(sub_values)} sub values: {sub_values}")

# Format bond values for display
def format_bond_value(bond_val):
    """Format bond value for display"""
    if bond_val == 0:
        return "$0k"
    elif bond_val < 1000000:
        thousands = bond_val // 1000
        return f"${thousands}k"
    else:
        millions = bond_val // 1000000
        remainder = (bond_val % 1000000) // 100000
        if remainder == 0:
            return f"${millions}.0M"
        else:
            return f"${millions}.{remainder}M"

# Format species name for display
def format_species_name(species_name):
    """Format species name for display"""
    if species_name == "B":
        return "B (500kg)"
    # Shorten long names
    if len(species_name) > 20:
        return species_name[:17] + "..."
    return species_name

# Create subplots
n_species = len(all_species)
n_cols = 3  # 3 columns
n_rows = (n_species + n_cols - 1) // n_cols  # Ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
if n_rows == 1:
    axes = axes.reshape(1, -1)
axes = axes.flatten()

# Create matrices and plot for each species
for species_idx, species_name in enumerate(all_species):
    ax = axes[species_idx]
    
    # Create matrix for this species
    matrix = np.full((len(bond_values), len(sub_values)), np.nan)
    baseline_umpy = baseline_umpy_by_species.get(species_name, 0.0)
    
    for i, bond_val in enumerate(bond_values):
        for j, sub_val in enumerate(sub_values):
            if (bond_val, sub_val) in species_data_dict[species_name]:
                umpy_value = species_data_dict[species_name][(bond_val, sub_val)]
                if baseline_umpy > 0:
                    percent_change = ((umpy_value - baseline_umpy) / baseline_umpy) * 100
                else:
                    percent_change = 0.0 if umpy_value == 0 else np.inf
                matrix[i, j] = percent_change
    
    # Create discrete colormap for percentage change
    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)
    
    # Create discrete color levels
    n_levels = 10
    levels = np.linspace(vmin, vmax, n_levels + 1)
    
    # Use a diverging colormap centered at 0% change
    cmap = plt.cm.RdYlGn_r
    norm = BoundaryNorm(levels, cmap.N)
    
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, norm=norm, 
                   interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(range(len(sub_values)))
    ax.set_xticklabels([f'{sv:.1f}' for sv in sub_values], fontsize=8)
    ax.set_yticks(range(len(bond_values)))
    ax.set_yticklabels([format_bond_value(bv) for bv in bond_values], fontsize=8)
    
    ax.set_xlabel('Substitution Rate', fontsize=10)
    ax.set_ylabel('Bond Amount', fontsize=10)
    ax.set_title(f'{format_species_name(species_name)}', fontsize=11, fontweight='bold')
    
    # Add text annotations in each cell
    for i in range(len(bond_values)):
        for j in range(len(sub_values)):
            value = matrix[i, j]
            if not np.isnan(value) and not np.isinf(value):
                # Format value for display
                text = f'{value:.1f}'
                normalized_value = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                text_color = 'white' if normalized_value < 0.3 or normalized_value > 0.7 else 'black'
                ax.text(j, i, text, ha='center', va='center', 
                       color=text_color, fontsize=7, fontweight='bold')
    
    # Add colorbar for this subplot
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('% Change', fontsize=8, rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=7)

# Hide unused subplots
for idx in range(n_species, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('UMPY % Change by Species (Relative to Bond $0k, Sub 0.0)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save plot
output_path = results_dir / "umpy_heatmap_by_species.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nHeatmap by species saved to: {output_path}")

plt.show()
