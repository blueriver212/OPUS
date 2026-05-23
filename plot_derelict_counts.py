import json
import os
import re
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

def extract_sub_value(folder_name):
    """Extract the Sub value from folder name like 'Bond_0_Sub_0.9_Life_5' -> 0.9"""
    match = re.search(r'Sub_([\d.]+)', folder_name)
    if match:
        return float(match.group(1))
    return None

def get_final_year_derelict_counts(species_data):
    """Get derelict counts for the final year only, separated by derelict class"""
    counts = {}
    
    for derelict_class in derelict_classes:
        counts[derelict_class] = 0.0
        if derelict_class in species_data:
            if FINAL_YEAR in species_data[derelict_class]:
                values = species_data[derelict_class][FINAL_YEAR]
                if isinstance(values, list):
                    counts[derelict_class] = sum(values)
                else:
                    counts[derelict_class] = values
    
    return counts

def get_total_derelict_count(species_data):
    """Sum all derelict counts across all derelict classes for final year"""
    counts = get_final_year_derelict_counts(species_data)
    return sum(counts.values())

# Collect data from all simulations
sub_values = []
derelict_counts = []
derelict_counts_by_class = defaultdict(list)  # {derelict_class: [counts for each sub_value]}

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
        
        # Calculate total derelict count for final year
        total_derelicts = get_total_derelict_count(species_data)
        
        # Get counts by derelict class
        counts_by_class = get_final_year_derelict_counts(species_data)
        
        sub_values.append(sub_value)
        derelict_counts.append(total_derelicts)
        
        # Store counts for each derelict class
        for derelict_class in derelict_classes:
            derelict_counts_by_class[derelict_class].append(counts_by_class[derelict_class])
        
        print(f"Sub_{sub_value}: Total derelicts = {total_derelicts:.2f}")
    
    except Exception as e:
        print(f"Error processing {species_data_file}: {e}")
        continue

# Sort by Sub value for proper plotting
sorted_data = sorted(zip(sub_values, derelict_counts))
sub_values_sorted, derelict_counts_sorted = zip(*sorted_data) if sorted_data else ([], [])

# Sort derelict counts by class to match sorted sub_values
sorted_indices = sorted(range(len(sub_values)), key=lambda i: sub_values[i])
for derelict_class in derelict_classes:
    derelict_counts_by_class[derelict_class] = [derelict_counts_by_class[derelict_class][i] for i in sorted_indices]

# Plot 1: Total derelict count
plt.figure(figsize=(10, 6))
plt.plot(sub_values_sorted, derelict_counts_sorted, marker='o', linewidth=2, markersize=8)
plt.xlabel('Sub Value', fontsize=12)
plt.ylabel('Total Derelict Count (Final Year)', fontsize=12)
plt.title(f'Total Derelict Count vs Sub Value ({FINAL_YEAR})', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add labels for each point
for sub_val, count in zip(sub_values_sorted, derelict_counts_sorted):
    plt.annotate(f'{sub_val}', 
                xy=(sub_val, count), 
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=9,
                alpha=0.7)

plt.tight_layout()

# Save plot 1
output_path1 = results_dir / "derelict_counts_total_plot.png"
plt.savefig(output_path1, dpi=300, bbox_inches='tight')
print(f"\nPlot 1 saved to: {output_path1}")

# Plot 2: Separate lines for each derelict species
plt.figure(figsize=(12, 7))
colors = plt.cm.tab10(range(len(derelict_classes)))

for i, derelict_class in enumerate(derelict_classes):
    counts = derelict_counts_by_class[derelict_class]
    plt.plot(sub_values_sorted, counts, marker='o', linewidth=2, markersize=6, 
             label=derelict_class, color=colors[i])

plt.xlabel('Sub Value', fontsize=12)
plt.ylabel('Derelict Count (Final Year)', fontsize=12)
plt.title(f'Derelict Count by Species vs Sub Value ({FINAL_YEAR})', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save plot 2
output_path2 = results_dir / "derelict_counts_by_species_plot.png"
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"Plot 2 saved to: {output_path2}")

plt.show()
