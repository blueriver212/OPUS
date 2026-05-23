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
        
        # Get all species that don't start with 'S'
        non_s_species = get_all_non_s_species(species_data)
        
        if len(non_s_species) == 0:
            print(f"Warning: No non-S species found in {subfolder.name}")
            continue
        
        # Calculate total UMPY across all non-S species
        total_umpy = 0.0
        
        for species_name in non_s_species:
            pop_ij_array = get_final_year_shell_data(species_data, species_name)
            mass_i = extract_mass_from_species_name(species_name)
            umpy = calculate_umpy(pop_ij_array, orbital_lifetimes, mass_i, t_sim, X)
            total_umpy += umpy
        
        data_dict[(bond_value, sub_value)] = total_umpy
        
        # Store baseline value (Bond_0_Sub_0)
        if bond_value == 0 and sub_value == 0.0:
            baseline_umpy = total_umpy
            print(f"Baseline (Bond_0_Sub_0): UMPY = {baseline_umpy:.6f}")
            print(f"  Found {len(non_s_species)} non-S species: {non_s_species}")
    
    except Exception as e:
        print(f"Error processing {species_data_files[0]}: {e}")
        import traceback
        traceback.print_exc()
        continue

if baseline_umpy is None:
    print("ERROR: Could not find baseline (Bond_0_Sub_0) value!")
    exit(1)

# Extract unique bond and sub values
bond_values = sorted(set(b for b, s in data_dict.keys()), reverse=True)
sub_values = sorted(set(s for b, s in data_dict.keys()))

print(f"\nFound {len(bond_values)} bond values: {bond_values}")
print(f"Found {len(sub_values)} sub values: {sub_values}")

# Create matrix for heatmap with percentage change
matrix = np.full((len(bond_values), len(sub_values)), np.nan)

for i, bond_val in enumerate(bond_values):
    for j, sub_val in enumerate(sub_values):
        if (bond_val, sub_val) in data_dict:
            umpy_value = data_dict[(bond_val, sub_val)]
            # Calculate percentage change relative to baseline
            percent_change = ((umpy_value - baseline_umpy) / baseline_umpy) * 100
            matrix[i, j] = percent_change

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

# Create heatmap
fig, ax = plt.subplots(figsize=(12, 10))

# Create discrete colormap for percentage change
vmin = np.nanmin(matrix)
vmax = np.nanmax(matrix)

# Create discrete color levels (e.g., 10 levels)
n_levels = 10
levels = np.linspace(vmin, vmax, n_levels + 1)

# Use a diverging colormap centered at 0% change
# RdYlGn_r: green=negative (good), yellow=zero, red=positive (bad)
cmap = plt.cm.RdYlGn_r
norm = BoundaryNorm(levels, cmap.N)

im = ax.imshow(matrix, aspect='auto', cmap=cmap, norm=norm, 
               interpolation='nearest')

# Set ticks and labels
ax.set_xticks(range(len(sub_values)))
ax.set_xticklabels([f'{sv:.1f}' for sv in sub_values])
ax.set_yticks(range(len(bond_values)))
ax.set_yticklabels([format_bond_value(bv) for bv in bond_values])

ax.set_xlabel('Substitution Rate', fontsize=14, fontweight='bold')
ax.set_ylabel('Bond Amount', fontsize=14, fontweight='bold')
ax.set_title('UMPY % Change (All Non-S Species) Relative to Bond $0k, Sub 0.0', 
             fontsize=16, fontweight='bold')

# Add discrete colorbar
cbar = plt.colorbar(im, ax=ax, boundaries=levels, ticks=levels[:-1] + np.diff(levels)/2)
cbar.set_label('% Change', fontsize=12, rotation=270, labelpad=20)
# Format colorbar ticks
cbar.ax.set_yticklabels([f'{tick:.1f}' for tick in levels[:-1] + np.diff(levels)/2])

# Add text annotations in each cell
for i in range(len(bond_values)):
    for j in range(len(sub_values)):
        value = matrix[i, j]
        if not np.isnan(value):
            # Format value for display (show 1 decimal place)
            text = f'{value:.1f}'
            # Choose text color based on background brightness
            normalized_value = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            # Use white text for darker backgrounds (low and high values), black for middle
            text_color = 'white' if normalized_value < 0.3 or normalized_value > 0.7 else 'black'
            ax.text(j, i, text, ha='center', va='center', 
                   color=text_color, fontsize=9, fontweight='bold')

plt.tight_layout()

# Save plot
output_path = results_dir / "umpy_heatmap_all_species_percent_change.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPercentage change heatmap saved to: {output_path}")

plt.show()
