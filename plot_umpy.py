import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Directory containing simulation results
results_dir = Path("Results/extensive_1_Bond_0_Sub_0")

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

def extract_sub_value(folder_name):
    """Extract the Sub value from folder name like 'Bond_0_Sub_0.9_Life_5' -> 0.9"""
    match = re.search(r'Sub_([\d.]+)', folder_name)
    if match:
        return float(match.group(1))
    return None

def extract_mass_from_species_name(species_name):
    """Extract mass from species name like 'N_521kg' -> 521.0, 'N_521.01kg' -> 521.01"""
    # Remove 'N_' prefix and 'kg' suffix, then convert to float
    match = re.search(r'N_([\d.]+)kg', species_name)
    if match:
        return float(match.group(1))
    return 0.0

def calculate_umpy(pop_ij_array, life_ij_array, mass_i, t_sim, X):
    """
    Calculate UMPY using the formula from the code:
    For each shell: umpy_factor = (exp(X * (life_ij / t_sim)) - 1) / (exp(X) - 1)
    Then: umpy_contribution = (mass_i * pop_ij * umpy_factor) / t_sim
    Total UMPY = sum over all shells
    
    Parameters:
    - pop_ij_array: array of population counts for each shell (20 values)
    - life_ij_array: array of orbital lifetimes for each shell (20 values)
    - mass_i: mass of the species (extracted from name)
    - t_sim: simulation time constant
    - X: exponent constant
    """
    n_shells = len(pop_ij_array)
    
    # Pre-calculate constant denominator
    e_X = np.exp(X)
    denominator = e_X - 1.0
    
    # Calculate the sum over all shells
    total_sum = 0.0
    for shell_idx in range(n_shells):
        pop_ij = pop_ij_array[shell_idx]
        life_ij = life_ij_array[shell_idx]
        
        # Calculate umpy_factor = (exp(X * (life_ij / t_sim)) - 1) / (exp(X) - 1)
        exponent = X * (life_ij / t_sim)
        exp_term = np.exp(exponent)
        umpy_factor = (exp_term - 1.0) / denominator
        
        # Calculate contribution from this shell: (mass_i * pop_ij * umpy_factor) / t_sim
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
        # Ensure we have exactly 20 values
        if len(values) == 20:
            return np.array(values)
        else:
            print(f"Warning: Expected 20 shells, got {len(values)} for {derelict_class}")
            # Pad with zeros or truncate as needed
            if len(values) < 20:
                return np.pad(values, (0, 20 - len(values)), 'constant')
            else:
                return np.array(values[:20])
    else:
        return np.zeros(20)

# Collect data from all simulations
sub_values = []
total_umpy_values = []
umpy_by_class = defaultdict(list)  # {derelict_class: [umpy values for each sub_value]}

# Loop through all subfolders
for subfolder in sorted(results_dir.iterdir()):
    if not subfolder.is_dir():
        continue
    
    # Extract Sub value from folder name
    sub_value = extract_sub_value(subfolder.name)
    if sub_value is None:
        print(f"Warning: Could not extract Sub value from {subfolder.name}")
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
        
        # Calculate UMPY for each derelict class
        umpy_per_class = {}
        total_umpy = 0.0
        
        for derelict_class in derelict_classes:
            # Get shell population data for this derelict class
            pop_ij_array = get_final_year_shell_data(species_data, derelict_class)
            
            # Extract mass from species name
            mass_i = extract_mass_from_species_name(derelict_class)
            
            # Calculate UMPY for this derelict class
            umpy = calculate_umpy(pop_ij_array, orbital_lifetimes, mass_i, t_sim, X)
            umpy_per_class[derelict_class] = umpy
            
            # Add to total
            total_umpy += umpy
        
        sub_values.append(sub_value)
        total_umpy_values.append(total_umpy)
        
        # Store UMPY for each derelict class
        for derelict_class in derelict_classes:
            umpy_by_class[derelict_class].append(umpy_per_class[derelict_class])
        
        print(f"Sub_{sub_value}: Total UMPY = {total_umpy:.6f}")
    
    except Exception as e:
        print(f"Error processing {species_data_file}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Sort by Sub value for proper plotting
sorted_data = sorted(zip(sub_values, total_umpy_values))
sub_values_sorted, total_umpy_sorted = zip(*sorted_data) if sorted_data else ([], [])

# Sort UMPY by class to match sorted sub_values
sorted_indices = sorted(range(len(sub_values)), key=lambda i: sub_values[i])
for derelict_class in derelict_classes:
    umpy_by_class[derelict_class] = [umpy_by_class[derelict_class][i] for i in sorted_indices]

# Plot 1: Total UMPY
plt.figure(figsize=(10, 6))
plt.plot(sub_values_sorted, total_umpy_sorted, marker='o', linewidth=2, markersize=8)
plt.xlabel('Sub Value', fontsize=12)
plt.ylabel('Total UMPY (Final Year)', fontsize=12)
plt.title(f'Total UMPY vs Sub Value ({FINAL_YEAR})', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add labels for each point
for sub_val, umpy_val in zip(sub_values_sorted, total_umpy_sorted):
    plt.annotate(f'{sub_val}', 
                xy=(sub_val, umpy_val), 
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=9,
                alpha=0.7)

plt.tight_layout()

# Save plot 1
output_path1 = results_dir / "umpy_total_plot.png"
plt.savefig(output_path1, dpi=300, bbox_inches='tight')
print(f"\nPlot 1 saved to: {output_path1}")

# Plot 2: UMPY for each derelict species
plt.figure(figsize=(12, 7))
colors = plt.cm.tab10(range(len(derelict_classes)))

for i, derelict_class in enumerate(derelict_classes):
    umpy_values = umpy_by_class[derelict_class]
    plt.plot(sub_values_sorted, umpy_values, marker='o', linewidth=2, markersize=6, 
             label=derelict_class, color=colors[i])

plt.xlabel('Sub Value', fontsize=12)
plt.ylabel('UMPY (Final Year)', fontsize=12)
plt.title(f'UMPY by Derelict Species vs Sub Value ({FINAL_YEAR})', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save plot 2
output_path2 = results_dir / "umpy_by_species_plot.png"
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"Plot 2 saved to: {output_path2}")

plt.show()
