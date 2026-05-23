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

# UMPY calculation parameters
t_sim = 100.0
X = 4.0
mass_i = 521.0  # Mass from N_521kg

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

def calculate_umpy_contributions(pop_ij_array, life_ij_array, mass_i, t_sim, X):
    """Calculate UMPY contribution from each shell"""
    n_shells = len(pop_ij_array)
    e_X = np.exp(X)
    denominator = e_X - 1.0
    
    contributions = np.zeros(n_shells)
    
    for shell_idx in range(n_shells):
        pop_ij = pop_ij_array[shell_idx]
        life_ij = life_ij_array[shell_idx]
        
        exponent = X * (life_ij / t_sim)
        exp_term = np.exp(exponent)
        umpy_factor = (exp_term - 1.0) / denominator
        
        shell_contribution = (mass_i * pop_ij * umpy_factor) / t_sim
        contributions[shell_idx] = shell_contribution
    
    return contributions

# Calculate UMPY factors for each shell (independent of population)
e_X = np.exp(X)
denominator = e_X - 1.0
umpy_factors = []
for life_ij in orbital_lifetimes:
    exponent = X * (life_ij / t_sim)
    exp_term = np.exp(exponent)
    umpy_factor = (exp_term - 1.0) / denominator
    umpy_factors.append(umpy_factor)
umpy_factors = np.array(umpy_factors)

# Collect data
sub_values = []
total_counts = []
total_umpy_values = []
shell_distributions = []
umpy_contributions_by_shell = []

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
        umpy_contributions = calculate_umpy_contributions(shell_data, orbital_lifetimes, mass_i, t_sim, X)
        total_umpy = np.sum(umpy_contributions)
        
        sub_values.append(sub_value)
        total_counts.append(total_count)
        total_umpy_values.append(total_umpy)
        shell_distributions.append(shell_data)
        umpy_contributions_by_shell.append(umpy_contributions)
        
        print(f"Sub_{sub_value}: Total count = {total_count:.2f}, Total UMPY = {total_umpy:.6f}")
    
    except Exception as e:
        print(f"Error processing {subfolder.name}: {e}")
        continue

# Sort by Sub value
sorted_indices = sorted(range(len(sub_values)), key=lambda i: sub_values[i])
sub_values_sorted = [sub_values[i] for i in sorted_indices]
total_counts_sorted = [total_counts[i] for i in sorted_indices]
total_umpy_sorted = [total_umpy_values[i] for i in sorted_indices]
shell_distributions_sorted = [shell_distributions[i] for i in sorted_indices]
umpy_contributions_sorted = [umpy_contributions_by_shell[i] for i in sorted_indices]

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Total counts vs Sub value
ax1 = axes[0, 0]
ax1.plot(sub_values_sorted, total_counts_sorted, marker='o', linewidth=2, markersize=8, label='Total Count')
ax1_twin = ax1.twinx()
ax1_twin.plot(sub_values_sorted, total_umpy_sorted, marker='s', linewidth=2, markersize=8, 
              color='red', label='Total UMPY')
ax1.set_xlabel('Sub Value', fontsize=12)
ax1.set_ylabel('Total Count', fontsize=12, color='blue')
ax1_twin.set_ylabel('Total UMPY', fontsize=12, color='red')
ax1.set_title(f'{derelict_class}: Count vs UMPY', fontsize=14, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='blue')
ax1_twin.tick_params(axis='y', labelcolor='red')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')

# Plot 2: UMPY contributions by shell for selected Sub values
ax2 = axes[0, 1]
selected_indices = [0, len(sub_values_sorted)//2, len(sub_values_sorted)-1]
colors = ['blue', 'green', 'red']
for idx, color in zip(selected_indices, colors):
    sub_val = sub_values_sorted[idx]
    contributions = umpy_contributions_sorted[idx]
    ax2.plot(altitude_midpoints, contributions, marker='o', linewidth=2, 
             markersize=6, label=f'Sub_{sub_val}', color=color)
ax2.set_xlabel('Altitude (km)', fontsize=12)
ax2.set_ylabel('UMPY Contribution', fontsize=12)
ax2.set_title(f'{derelict_class} UMPY Contributions by Shell', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Shell distribution comparison
ax3 = axes[1, 0]
for idx, color in zip(selected_indices, colors):
    sub_val = sub_values_sorted[idx]
    shell_data = shell_distributions_sorted[idx]
    ax3.plot(altitude_midpoints, shell_data, marker='o', linewidth=2, 
             markersize=6, label=f'Sub_{sub_val}', color=color, alpha=0.7)
ax3.set_xlabel('Altitude (km)', fontsize=12)
ax3.set_ylabel(f'{derelict_class} Count', fontsize=12)
ax3.set_title(f'{derelict_class} Distribution by Altitude', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: UMPY factor vs orbital lifetime (showing why high-altitude shells matter more)
ax4 = axes[1, 1]
ax4.plot(orbital_lifetimes, umpy_factors, marker='o', linewidth=2, markersize=8)
ax4.set_xlabel('Orbital Lifetime (years)', fontsize=12)
ax4.set_ylabel('UMPY Factor', fontsize=12)
ax4.set_title('UMPY Factor vs Orbital Lifetime', fontsize=14, fontweight='bold')
ax4.set_xscale('log')
ax4.grid(True, alpha=0.3)
# Add annotations for key shells
for i in [0, 6, 10, 15, 19]:
    ax4.annotate(f'Shell {i}\n({altitude_midpoints[i]} km)', 
                xy=(orbital_lifetimes[i], umpy_factors[i]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

plt.tight_layout()

# Save plot
output_path = results_dir / f"{derelict_class}_umpy_contributions_analysis.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nAnalysis plot saved to: {output_path}")

# Print detailed comparison
print("\n" + "="*80)
print(f"Detailed Comparison for {derelict_class}:")
print("="*80)
print(f"{'Sub':<8} {'Total':<12} {'UMPY':<12} {'Top 3 Shells Contributing to UMPY':<50}")
print("-"*80)

for i, (sub_val, total, umpy, contributions) in enumerate(zip(
    sub_values_sorted, total_counts_sorted, total_umpy_sorted, umpy_contributions_sorted)):
    
    # Find top 3 shells contributing to UMPY
    top3_indices = np.argsort(contributions)[-3:][::-1]
    top3_str = ", ".join([f"Shell {idx} ({altitude_midpoints[idx]:.0f}km, {contributions[idx]:.4f})" 
                          for idx in top3_indices])
    
    print(f"{sub_val:<8.1f} {total:<12.2f} {umpy:<12.6f} {top3_str}")

plt.show()
