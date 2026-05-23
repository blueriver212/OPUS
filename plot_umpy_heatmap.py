import json
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
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

def extract_bond_sub_values(folder_name):
    """Extract Bond and Sub values from folder name like 'Bond_100000_Sub_0.5_Life_5'"""
    match = re.search(r'Bond_([\d]+)_Sub_([\d.]+)', folder_name)
    if match:
        bond_value = int(match.group(1))
        sub_value = float(match.group(2))
        return bond_value, sub_value
    return None, None

def extract_mass_from_species_name(species_name):
    """Extract mass from species name like 'N_521kg' -> 521.0"""
    match = re.search(r'N_([\d.]+)kg', species_name)
    if match:
        return float(match.group(1))
    return 0.0

def calculate_umpy(pop_ij_array, life_ij_array, mass_i, t_sim, X):
    """Calculate UMPY for a derelict species"""
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

# Collect data from all simulations
data_dict = {}  # {(bond, sub): umpy_value}

for subfolder in sorted(results_dir.iterdir()):
    if not subfolder.is_dir():
        continue
    
    # Extract Bond and Sub values
    bond_value, sub_value = extract_bond_sub_values(subfolder.name)
    if bond_value is None or sub_value is None:
        continue
    
    # Find species_data JSON file
    species_data_files = list(subfolder.glob("species_data_*.json"))
    if not species_data_files:
        print(f"Warning: No species_data JSON found in {subfolder.name}")
        continue
    
    species_data_file = species_data_files[0]
    
    # Load JSON data
    try:
        with open(species_data_file, 'r') as f:
            species_data = json.load(f)
        
        # Calculate total UMPY across all derelict classes
        total_umpy = 0.0
        
        for derelict_class in derelict_classes:
            pop_ij_array = get_final_year_shell_data(species_data, derelict_class)
            mass_i = extract_mass_from_species_name(derelict_class)
            umpy = calculate_umpy(pop_ij_array, orbital_lifetimes, mass_i, t_sim, X)
            total_umpy += umpy
        
        data_dict[(bond_value, sub_value)] = total_umpy
        print(f"Bond_{bond_value}_Sub_{sub_value}: UMPY = {total_umpy:.6f}")
    
    except Exception as e:
        print(f"Error processing {species_data_file}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Extract unique bond and sub values
# Reverse bond values so smaller bonds are at bottom
bond_values = sorted(set(b for b, s in data_dict.keys()), reverse=True)
sub_values = sorted(set(s for b, s in data_dict.keys()))

print(f"\nFound {len(bond_values)} bond values: {bond_values}")
print(f"Found {len(sub_values)} sub values: {sub_values}")

# Create matrix for heatmap
matrix = np.full((len(bond_values), len(sub_values)), np.nan)

for i, bond_val in enumerate(bond_values):
    for j, sub_val in enumerate(sub_values):
        if (bond_val, sub_val) in data_dict:
            matrix[i, j] = data_dict[(bond_val, sub_val)]

# Format bond values for display (convert to thousands or millions)
def format_bond_value(bond_val):
    """Format bond value for display to match reference style"""
    if bond_val == 0:
        return "$0k"
    elif bond_val < 1000000:
        # Convert to thousands
        thousands = bond_val // 1000
        return f"${thousands}k"
    else:
        # Convert to millions with one decimal place
        millions = bond_val // 1000000
        remainder = (bond_val % 1000000) // 100000
        if remainder == 0:
            return f"${millions}.0M"
        else:
            return f"${millions}.{remainder}M"

# Create heatmap
fig, ax = plt.subplots(figsize=(12, 10))

# Create heatmap with discrete colormap
vmin = np.nanmin(matrix)
vmax = np.nanmax(matrix)

# Create discrete color levels (e.g., 10 levels)
n_levels = 10
levels = np.linspace(vmin, vmax, n_levels + 1)

# Use a sequential colormap that's intuitive:
# - Lower UMPY (better) = lighter colors (green/yellow)
# - Higher UMPY (worse) = darker colors (red/orange)
# Options: 'RdYlGn_r' (red-yellow-green reversed), 'YlOrRd' (yellow-orange-red), 
# 'viridis_r', 'plasma_r', or 'RdYlGn_r' for intuitive good/bad
# Using RdYlGn_r: green=low (good), yellow=medium, red=high (bad)
cmap = plt.cm.RdYlGn_r  # Red-Yellow-Green reversed: green=low, red=high
norm = BoundaryNorm(levels, cmap.N)

# Use nearest interpolation for discrete colors
im = ax.imshow(matrix, aspect='auto', cmap=cmap, norm=norm, 
               interpolation='nearest')

# Set ticks and labels
ax.set_xticks(range(len(sub_values)))
ax.set_xticklabels([f'{sv:.1f}' for sv in sub_values])
ax.set_yticks(range(len(bond_values)))
ax.set_yticklabels([format_bond_value(bv) for bv in bond_values])

ax.set_xlabel('Substitution Rate', fontsize=14, fontweight='bold')
ax.set_ylabel('Bond Amount', fontsize=14, fontweight='bold')
ax.set_title('Total Derelict UMPY (Final Year)', fontsize=16, fontweight='bold')

# Add discrete colorbar
cbar = plt.colorbar(im, ax=ax, boundaries=levels, ticks=levels[:-1] + np.diff(levels)/2)
cbar.set_label('UMPY Value', fontsize=12, rotation=270, labelpad=20)
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
            # For RdYlBu_r: darker colors are at extremes, lighter in middle
            normalized_value = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            # Use white text for darker backgrounds (low and high values), black for middle
            text_color = 'white' if normalized_value < 0.3 or normalized_value > 0.7 else 'black'
            ax.text(j, i, text, ha='center', va='center', 
                   color=text_color, fontsize=9, fontweight='bold')

plt.tight_layout()

# Save plot
output_path = results_dir / "umpy_heatmap.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nHeatmap saved to: {output_path}")

plt.show()
