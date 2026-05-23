import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Directory containing simulation results
results_dir = Path("Results/extensive_1_Bond_0_Sub_0")

# Derelict class to analyze
derelict_class = "N_521kg"
FINAL_YEAR = "2049"

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
    """Extract the Sub value from folder name"""
    match = re.search(r'Sub_([\d.]+)', folder_name)
    if match:
        return float(match.group(1))
    return None

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

# Collect data
sub_values = []
total_counts = []
shell_distributions = []  # List of arrays, one per sub_value

for subfolder in sorted(results_dir.iterdir()):
    if not subfolder.is_dir():
        continue
    
    sub_value = extract_sub_value(subfolder.name)
    if sub_value is None:
        continue
    
    species_data_files = list(subfolder.glob("species_data_*.json"))
    if not species_data_files:
        continue
    
    try:
        with open(species_data_files[0], 'r') as f:
            species_data = json.load(f)
        
        shell_data = get_final_year_shell_data(species_data, derelict_class)
        total_count = np.sum(shell_data)
        
        sub_values.append(sub_value)
        total_counts.append(total_count)
        shell_distributions.append(shell_data)
        
        print(f"Sub_{sub_value}: Total {derelict_class} = {total_count:.2f}")
    
    except Exception as e:
        print(f"Error processing {subfolder.name}: {e}")
        continue

# Sort by Sub value
sorted_indices = sorted(range(len(sub_values)), key=lambda i: sub_values[i])
sub_values_sorted = [sub_values[i] for i in sorted_indices]
total_counts_sorted = [total_counts[i] for i in sorted_indices]
shell_distributions_sorted = [shell_distributions[i] for i in sorted_indices]

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Total counts vs Sub value
ax1 = axes[0, 0]
ax1.plot(sub_values_sorted, total_counts_sorted, marker='o', linewidth=2, markersize=8)
ax1.set_xlabel('Sub Value', fontsize=12)
ax1.set_ylabel(f'Total {derelict_class} Count', fontsize=12)
ax1.set_title(f'Total {derelict_class} Count vs Sub Value', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
for sub_val, count in zip(sub_values_sorted, total_counts_sorted):
    ax1.annotate(f'{sub_val}', xy=(sub_val, count), xytext=(5, 5), 
                textcoords='offset points', fontsize=8, alpha=0.7)

# Plot 2: Shell distribution heatmap
ax2 = axes[0, 1]
# Create a matrix: rows = sub_values, columns = shells
distribution_matrix = np.array(shell_distributions_sorted)
im = ax2.imshow(distribution_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
ax2.set_xlabel('Shell Index', fontsize=12)
ax2.set_ylabel('Sub Value Index', fontsize=12)
ax2.set_title(f'{derelict_class} Distribution Across Shells', fontsize=14, fontweight='bold')
ax2.set_yticks(range(len(sub_values_sorted)))
ax2.set_yticklabels([f'{sv:.1f}' for sv in sub_values_sorted])
plt.colorbar(im, ax=ax2, label='Count')

# Plot 3: Distribution for a few selected Sub values
ax3 = axes[1, 0]
# Select a few representative Sub values
selected_indices = [0, len(sub_values_sorted)//2, len(sub_values_sorted)-1]
for idx in selected_indices:
    sub_val = sub_values_sorted[idx]
    shell_data = shell_distributions_sorted[idx]
    ax3.plot(altitude_midpoints, shell_data, marker='o', linewidth=2, 
             markersize=6, label=f'Sub_{sub_val}')
ax3.set_xlabel('Altitude (km)', fontsize=12)
ax3.set_ylabel(f'{derelict_class} Count', fontsize=12)
ax3.set_title(f'{derelict_class} Distribution by Altitude', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Weighted contribution (UMPY factor * count) vs shell
ax4 = axes[1, 1]
t_sim = 100.0
X = 4.0
e_X = np.exp(X)
denominator = e_X - 1.0

# Calculate UMPY factor for each shell
umpy_factors = []
for life_ij in orbital_lifetimes:
    exponent = X * (life_ij / t_sim)
    exp_term = np.exp(exponent)
    umpy_factor = (exp_term - 1.0) / denominator
    umpy_factors.append(umpy_factor)

umpy_factors = np.array(umpy_factors)

# Show weighted contribution for selected Sub values
for idx in selected_indices:
    sub_val = sub_values_sorted[idx]
    shell_data = shell_distributions_sorted[idx]
    weighted = shell_data * umpy_factors
    ax4.plot(altitude_midpoints, weighted, marker='o', linewidth=2, 
             markersize=6, label=f'Sub_{sub_val} (weighted)')

ax4.set_xlabel('Altitude (km)', fontsize=12)
ax4.set_ylabel(f'Weighted Contribution (count × UMPY factor)', fontsize=12)
ax4.set_title(f'{derelict_class} UMPY-Weighted Distribution', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save plot
output_path = results_dir / f"{derelict_class}_shell_distribution_analysis.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nAnalysis plot saved to: {output_path}")

# Print summary statistics
print("\n" + "="*60)
print(f"Summary for {derelict_class}:")
print("="*60)
print(f"{'Sub Value':<12} {'Total Count':<15} {'Shell with Max':<20} {'Max Shell Alt':<15}")
print("-"*60)
for i, (sub_val, total, shell_data) in enumerate(zip(sub_values_sorted, total_counts_sorted, shell_distributions_sorted)):
    max_shell_idx = np.argmax(shell_data)
    max_shell_alt = altitude_midpoints[max_shell_idx]
    print(f"{sub_val:<12.1f} {total:<15.2f} {max_shell_idx:<20} {max_shell_alt:<15.1f}")

plt.show()
