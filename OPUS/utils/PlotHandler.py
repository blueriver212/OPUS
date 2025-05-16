import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import math
import pickle

class PlotData:
        """
                This will hold in the data required to make any of the plots. 
                It will also be passed to each of the plotting functions.       
        """
        def __init__(self, scenario, path, MOCAT):
                self.scenario = scenario
                self.path = path

                # Load the data from the scenario folder
                self.data, self.other_data, self.econ_params = self.load_data(path)
                self.species_names = MOCAT.scenario_properties.species_names
                self.n_shells = MOCAT.scenario_properties.n_shells
                self.n_species = MOCAT.scenario_properties.species_length
                self.HMid = MOCAT.scenario_properties.HMid


        def get_other_data(self):
                return self.other_data

        def load_data(self, path):
                """
                Load the data from the scenario folder, prioritizing files with 'species_data' and 'other_results'
                """
                if not os.path.exists(path):
                        print(f"Error: {path} does not exist.")
                        return None
                
                # Find the JSON files in the folder
                json_files = [f for f in os.listdir(path) if f.endswith(".json")]

                if len(json_files) == 0:
                        print(f"Error: No JSON file found in {path}.")
                        return None
                
                # Initialize variables for storing the two data files
                data = None
                other_data = None
                econ_params = None
                
                # Loop through the files and assign them based on content
                for file in json_files:
                        file_path = os.path.join(path, file)
                        with open(file_path, "r") as f:
                                file_content = json.load(f)
                                
                                if "species_data" in file_path:
                                        data = file_content  # First, assign the species data
                                elif "other_results" in file_path:
                                        other_data = file_content  # Then, assign the other results data
                                elif "econ_params" in file_path:
                                        econ_params = file_content

                if data is None:
                        print(f"Error: No file containing 'species_data' found in {path}.")
                if other_data is None:
                        print(f"Error: No file containing 'other_results' found in {path}.")
                if econ_params is None:
                        print(f"Error: No file containing 'econ_params' found in {path}.")
                
                return data, other_data, econ_params


class PlotHandler:  
        def __init__(self, MOCAT, scenario_files, simulation_name, plot_types=["all_plots"], comparison=True):
                """
                Initialize the PlotHandler.
                Comparison will compare all of the simulation names
                """
                
                self.MOCAT = MOCAT
                self.scenario_files = scenario_files # This will be a list of each sub-scenario run name
                self.simulation_name = simulation_name # This is the overall name of the simualtion 
                self.plot_types = plot_types # This will be a list of the types of plots to be generated
                self.HMid = self.MOCAT.scenario_properties.HMid
                self.n_shells = self.MOCAT.scenario_properties.n_shells

                # This will rely on the fact that there is a file available under the simulation name in the Results folder. 
                self.simulation_folder = os.path.join("Results", self.simulation_name)
                
                # if not show error to the user
                if not os.path.exists(self.simulation_folder):
                        print(f"Error: {self.simulation_folder} does not exist.")
                        return
                
                plot_data_list = []
                other_data_list = []
                econ_params_list = []

                # Loop through the scenario files and generate the plots
                for scenario in self.scenario_files:
                        scenario_folder = os.path.join(self.simulation_folder, scenario)
                        if not os.path.exists(scenario_folder):
                                print(f"Error: {scenario_folder} folder does not exist. Skipping scenario...")
                                continue
                        else: 
                                print("Generating plots for scenario: ", scenario)

                                # Build a PlotData object and then pass to the plotting functions
                                plot_data = PlotData(scenario, scenario_folder, MOCAT)
                                other_data = plot_data.get_other_data()
                                econ_data = plot_data.econ_params

                                # Add to lists for comparison plots
                                plot_data_list.append(plot_data)
                                other_data_list.append(other_data)
                                econ_params_list.append(econ_data)

                                # If the plot_types is None, then generate all plots
                                if "all_plots" in self.plot_types:
                                        self.all_plots(plot_data, other_data, econ_data)
                                else:
                                        # Dynamically generate plots
                                        for plot_name in self.plots:
                                                plot_method = getattr(self, plot_name, None)
                                                if callable(plot_method):
                                                        print(f"Creating plot: {plot_name}")
                                                        plot_method()
                                                else:
                                                        print(f"Warning: Plot '{plot_name}' not found. Skipping...")

                if comparison:
                        self._comparison_plots(plot_data_list, other_data_list)
                
        def _comparison_plots(self, plot_data_lists, other_data_lists):
                """
                Run all plot functions that start with 'comparison_', ignoring others.
                """
                for attr_name in dir(self):
                        # Grab the attribute; see if it’s a callable (method)
                        attr = getattr(self, attr_name)
                        if callable(attr):
                                # Skip known special methods
                                if attr_name in ("__init__", "all_plots"):
                                        continue

                                # Only call if it starts with 'comparison_'
                                if attr_name.startswith("comparison_"):
                                        print(f"Creating plot: {attr_name}")
                                        try:
                                                plot_method = attr
                                                plot_method(plot_data_lists, other_data_lists)
                                        except Exception as e:
                                                print(f"⚠️ Failed to generate plot '{attr_name}': {e}")

        def all_plots(self, plot_data, other_data, econ_params):
                """
                Run all plot functions, irrespective of the plots list.
                """
                for attr_name in dir(self):
                        if callable(getattr(self, attr_name)) and attr_name not in ("__init__", "all_plots"):
                                if not attr_name.startswith("_") and not attr_name.startswith("comparison_") and not attr_name.startswith("econ_"):
                                        print(f"Creating plot: {attr_name}")
                                        plot_method = getattr(self, attr_name)
                                        plot_method(plot_data, other_data)
                                elif attr_name.startswith("econ_"):
                                        print(f"Creating plot: {attr_name}")
                                        plot_method = getattr(self, attr_name)
                                        plot_method(plot_data.path, econ_params)


        def econ_create_individual_plot_of_params(self, path, econ_params):
                """
                Create individual plots for economic metrics based on a user-defined list.
                Each entry in metrics_info is a dictionary with the following keys:
                - "metric_key": the key to look up in econ_params
                - "y_label": desired label for the y-axis (if not found, defaults to "Value")
                - "file_name": desired file name for the plot (if not found, defaults to "{metric_key}.png")
                
                The plot title will use the provided y_label; if not provided, it will use the JSON key.
                Only metrics whose values are lists and match the expected number of shells are plotted.
                """
                metrics_info = [
                        {"metric_key": "cost", "y_label": "Total Cost", "file_name": "total_cost.png"},
                        {"metric_key": "total_deorbit_delta_v", "y_label": "Total Δv for Deorbit", "file_name": "total_deorbit_delta_v.png"},
                        {"metric_key": "lifetime_loss_cost", "y_label": "Lifetime Loss Cost", "file_name": "lifetime_loss_cost.png"},
                        {"metric_key": "stationkeeping_cost", "y_label": "Stationkeeping Cost", "file_name": "stationkeeping_cost.png"},
                        {"metric_key": "v_drag", "y_label": "Δv Required to Counter Drag", "file_name": "v_drag.png"},
                        {"metric_key": "lifetime_after_deorbit", "y_label": "Lifetime After Deorbit", "file_name": "lifetime_after_deorbit.png"},
                        {"metric_key": "delta_v_after_deorbit", "y_label": "Δv Leftover After Deorbit", "file_name": "delta_v_after_deorbit.png"}
                ]

                # Create a new econ folder for saving the plots.
                econ_folder = os.path.join(path, "econ_params")
                os.makedirs(econ_folder, exist_ok=True)
                
                # Retrieve shell mid-altitudes and the expected number of shells.
                shell_mid_altitudes = self.HMid
                n_shells = self.n_shells

                # Loop through each metric specification.
                for item in metrics_info:
                        metric_key = item.get("metric_key")
                        y_label = item.get("y_label", "Value")
                        file_name = item.get("file_name", f"{metric_key}.png")
                        file_path = os.path.join(econ_folder, file_name)
                        
                        # Retrieve the metric value from the JSON dictionary.
                        metric_value = econ_params.get(metric_key)
                        if metric_value is None:
                                print(f"Warning: '{metric_key}' not found in economic parameters. Skipping plot.")
                                continue
                        if not isinstance(metric_value, list):
                                print(f"Warning: Value for '{metric_key}' is not a list. Skipping plot.")
                                continue
                        if len(metric_value) != n_shells:
                                print(f"Warning: Length of '{metric_key}' ({len(metric_value)}) does not match number of shells ({n_shells}). Skipping plot.")
                                continue
                        
                        # Create the plot.
                        plt.figure(figsize=(8, 6))
                        plt.plot(shell_mid_altitudes, metric_value, marker='o', linestyle='-')
                        plt.xlabel("Shell Mid Altitude (km)")
                        plt.ylabel(y_label)
                        # Title uses y_label if provided; otherwise, it falls back to the JSON key.
                        plt.title(f"{y_label} vs. Shell Mid Altitude [{metric_key}]")
                        plt.xticks(shell_mid_altitudes, rotation=45)
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig(file_path)
                        plt.close()
                        print(f"Plot for '{metric_key}' saved to {file_path}")

        def econ_all_on_one_plot_line(self, path, econ_params):
                """
                Creates a single composite plot with all selected economic metrics.
                Each metric is plotted as a separate line on the same axes.

                Parameters
                ----------
                econ_params : dict
                        Dictionary containing economic parameters loaded from JSON.
                metrics_info : list
                        A list of dictionaries where each dictionary specifies:
                        - "metric_key": The key in econ_params to plot.
                        - "y_label": (Optional) The label to use in the legend for this metric.
                                If not provided, the metric_key will be used.
                        - "file_name": (Optional) A desired file name for an individual plot (ignored in this function).
                file_name : str, optional
                        Name of the saved composite figure file, by default 'all_metrics_single_plot.png'.
                """

                metrics_info = [
                        # {"metric_key": "cost", "y_label": "Total Cost", "file_name": "total_cost.png"},
                        {"metric_key": "total_deorbit_delta_v", "y_label": "Total Δv for Deorbit", "file_name": "total_deorbit_delta_v.png"},
                        {"metric_key": "lifetime_loss_cost", "y_label": "Lifetime Loss Cost", "file_name": "lifetime_loss_cost.png"},
                        {"metric_key": "stationkeeping_cost", "y_label": "Stationkeeping Cost", "file_name": "stationkeeping_cost.png"},
                        {"metric_key": "v_drag", "y_label": "Δv Required to Counter Drag", "file_name": "v_drag.png"},
                        {"metric_key": "lifetime_after_deorbit", "y_label": "Lifetime After Deorbit", "file_name": "lifetime_after_deorbit.png"},
                        {"metric_key": "delta_v_after_deorbit", "y_label": "Δv Leftover After Deorbit", "file_name": "delta_v_after_deorbit.png"}
                ]

                # Create the econ_params folder
                econ_folder = os.path.join(path, "econ_params")
                os.makedirs(econ_folder, exist_ok=True)
                file_path = os.path.join(econ_folder, "all_econ_params_one_plot.png")
                
                # Retrieve shell mid-altitudes and expected number of shells
                n_shells = self.n_shells
                shell_mid_altitudes = self.HMid
                
                plt.figure(figsize=(10, 6))
                
                # Loop through the metrics_info list and plot each metric if it exists and is valid.
                for item in metrics_info:
                        metric_key = item.get("metric_key")
                        y_label = item.get("y_label", metric_key)  # Use provided label or default to metric_key
                        
                        # Retrieve the metric from econ_params.
                        metric_value = econ_params.get(metric_key)
                        if metric_value is None:
                                print(f"Warning: '{metric_key}' not found in economic parameters. Skipping.")
                                continue
                        if not isinstance(metric_value, list):
                                print(f"Warning: Value for '{metric_key}' is not a list. Skipping.")
                                continue
                        if len(metric_value) != n_shells:
                                print(f"Warning: Length of '{metric_key}' ({len(metric_value)}) does not match number of shells ({n_shells}). Skipping.")
                                continue
                                
                        plt.plot(shell_mid_altitudes, metric_value, marker='o', linestyle='-', label=y_label)
                
                plt.xlabel("Shell Mid Altitude (km)")
                plt.ylabel("Value")
                plt.title("Economic Metrics Comparison")
                plt.legend()
                plt.xticks(shell_mid_altitudes, rotation=45)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(file_path)
                plt.close()
                print(f"All economic metrics single plot saved to {file_path}")

        def econ_costs_stacked_bar_chart(self, path, econ_params):
                """
                Creates a stacked bar chart of selected economic metrics for each altitude.
                Each bar (for a given shell mid-altitude) is divided into segments corresponding 
                to different economic metrics.
                
                Parameters
                ----------
                path : str
                        Base path where the econ_params folder will be created.
                econ_params : dict
                        Dictionary containing economic parameters loaded from JSON.
                """
                # Define the metrics_info list that can be edited manually.
                metrics_info = [
                        {"metric_key": "total_lift_price", "y_label": "Total Lift Price ($)", "file_name": "total_lift_price.png"},
                        {"metric_key": "stationkeeping_cost", "y_label": "Stationkeeping Cost ($)", "file_name": "stationkeeping_cost.png"},
                        {"metric_key": "lifetime_loss_cost", "y_label": "Lifetime Loss Cost ($)", "file_name": "lifetime_loss_cost.png"},
                        {"metric_key": "deorbit_maneuver_cost", "y_label": "Deorbit Maneuver  ($)", "file_name": "deorbit_maneuver_cost.png"},
                        {"metric_key": "bstar", "y_label": "Bond Amount ($)", "file_name": "bstar.png"}
                ]
                
                # Create the econ_params folder for saving the plot.
                econ_folder = os.path.join(path, "econ_params")
                os.makedirs(econ_folder, exist_ok=True)
                file_path = os.path.join(econ_folder, "costs_per_shell_stacked_bar.png")
                
                # Retrieve the expected number of shells and the corresponding mid-altitudes.
                n_shells = self.n_shells
                shell_mid_altitudes = self.HMid  # Expected to be a list (or array) of numeric altitudes
                
                # Prepare lists to store valid metric arrays and their labels.
                valid_metrics = []
                valid_labels = []
                
                # Loop through the metrics_info and validate each metric.
                for item in metrics_info:
                        metric_key = item.get("metric_key")
                        y_label = item.get("y_label", metric_key)
                        
                        # Retrieve the metric from econ_params.
                        metric_value = econ_params.get(metric_key)
                        if metric_value is None:
                                print(f"Warning: '{metric_key}' not found in economic parameters. Skipping.")
                                continue
                        if not isinstance(metric_value, list):
                                print(f"Warning: Value for '{metric_key}' is not a list. Skipping.")
                                continue
                        if len(metric_value) != n_shells:
                                print(f"Warning: Length of '{metric_key}' ({len(metric_value)}) does not match number of shells ({n_shells}). Skipping.")
                                continue
                        
                        valid_metrics.append(metric_value)
                        valid_labels.append(y_label)
                
                if not valid_metrics:
                        print("No valid economic metric lists found to plot.")
                        return
                
                # Convert the list of valid metrics into a NumPy array with shape (num_metrics, n_shells)
                import numpy as np
                valid_metrics_array = np.array(valid_metrics)  # each row corresponds to one metric
                
                # Use the shell mid-altitudes as the x positions. Ensure it's a NumPy array.
                x = np.array(shell_mid_altitudes)
                
                # Create the stacked bar chart.
                plt.figure(figsize=(10, 6))
                
                # Initialize the bottom of the bars at zero.
                bottom = np.zeros(n_shells)
                
                # For each metric, plot a bar segment at each altitude.
                for metric, label in zip(valid_metrics_array, valid_labels):
                        # Here we use plt.bar with the 'bottom' argument to stack the bars.
                        # The width is set relative to the distance between altitudes (if more than one exists).
                        width = (x[1] - x[0]) * 0.8 if len(x) > 1 else 0.5
                        plt.bar(x, metric, width=width, bottom=bottom, label=label)
                        bottom += metric  # Update the bottom for the next metric's bars
                
                plt.xlabel("Shell Mid Altitude (km)")
                plt.ylabel("Value")
                plt.title("Stacked Economic Metrics by Altitude")
                plt.legend()
                plt.xticks(x, rotation=45)
                plt.grid(True, axis='y')
                plt.tight_layout()
                plt.savefig(file_path)
                plt.close()
                print(f"Stacked bar chart saved to {file_path}")

        
        def comparison_total_species_count(self, plot_data_lists, other_data_lists):
                """
                Creates a comparison plot of total species count over time.
                Each species is plotted in its own subplot, comparing across all scenarios.
                """

                # Create a "comparisons" folder under the main simulation folder
                comparison_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(comparison_folder, exist_ok=True)

                # Dictionary to store time series data for each species across scenarios
                # Structure: species_totals[species][scenario_name] = np.array of total counts over time
                species_totals = {}

                # Loop over each PlotData to extract data
                for i, plot_data in enumerate(plot_data_lists):
                        # We assume `plot_data.scenario` holds the scenario name
                        # If it doesn't, change the attribute name or use a default label.
                        scenario_name = getattr(plot_data, "scenario", f"Scenario {i+1}")

                        # Retrieve the dictionary of species -> data arrays
                        # e.g., {species: np.array(time, shells), ...}
                        data_dict = plot_data.data

                        for species, species_data in data_dict.items():
                        # Sum across shells to get a total count per time step
                        # species_data has shape (time, shells), so we sum across axis=1
                                total_species_count = np.sum(species_data, axis=1)  # shape: (time,)

                                # Store data per species
                                if species not in species_totals:
                                        species_totals[species] = {}

                                # Keep track of the total count array by scenario
                                species_totals[species][scenario_name] = total_species_count

                # Count how many species we have
                num_species = len(species_totals)

                print(num_species)
                print("Species found overall:", list(species_totals.keys()))

                # If multiple species, create subplots in a grid
                num_cols = 2
                num_rows = math.ceil(num_species / num_cols)

                fig, axes = plt.subplots(
                        nrows=num_rows,
                        ncols=num_cols,
                        figsize=(12, 6 * num_rows),
                        sharex=True
                )

                # Flatten axes for easy iteration (in case num_rows > 1)
                axes = np.array(axes).flatten()

                # Plot each species in its own subplot
                for idx, (species, scenario_data) in enumerate(species_totals.items()):
                        ax = axes[idx]
                        # scenario_data looks like {scenario_name: np.array([...])}
                        for scenario_name, counts in scenario_data.items():
                                ax.plot(range(len(counts)), counts, label=scenario_name, marker='o')

                        ax.set_title(f"Total Count across all shells: {species}")
                        ax.set_xlabel("Time Steps (or Years)")
                        ax.set_ylabel("Total Count")
                        ax.legend()
                        ax.grid(True)

                # Hide any leftover empty subplots (if #species < num_rows * num_cols)
                for extra_ax in axes[num_species:]:
                        extra_ax.set_visible(False)

                plt.tight_layout()

                # Save the figure
                out_path = os.path.join(comparison_folder, "comparison_species_count.png")
                plt.savefig(out_path, dpi=300)
                plt.close()

                print(f"Comparison plot saved to {out_path}")

        def comparison_UMPY(self, plot_data_lists, other_data_lists):
                """
                Create a comparison plot of total UMPY over time for multiple scenarios.
                Each scenario is plotted on the same figure with a label derived from 
                its scenario name.
                """
                comparison_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(comparison_folder, exist_ok=True)

                # Create a single figure for all scenarios
                plt.figure(figsize=(8, 5))

                # Loop through each plot_data and other_data pair
                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        # 1) Sort the timesteps
                        timesteps = sorted(other_data.keys(), key=int)
                        umpy_sums = []

                        # 2) Sum the 'umpy' values for each timestep
                        for ts in timesteps:
                                umpy_list = other_data[ts]["umpy"]  # This is assumed to be a list of floats
                                total_umpy = np.sum(umpy_list)
                                umpy_sums.append(total_umpy)

                        # Here we assume `plot_data` has an attribute storing the scenario name.
                        # Adjust this to match your actual code if the attribute differs.
                        scenario_label = getattr(plot_data, 'scenario', f"Scenario {i+1}")

                        # 3) Plot each scenario on the same figure
                        plt.plot(
                        timesteps,
                        umpy_sums,
                        marker='o',
                        label=scenario_label
                        )

                # 4) Labels, legend, and layout
                plt.xlabel("Year (timestep)")
                plt.ylabel("UMPY (kg/year)")
                plt.title("UMPY Evolution Over Time (All Scenarios)")
                plt.legend()
                plt.tight_layout()

                # 5) Save the figure using the first plot_data's path 
                out_path = os.path.join(comparison_folder, "umpy_over_time.png")
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Comparison UMPY plot saved to {out_path}")

        def comparison_scatter_noncompliance_vs_bond(self, plot_data_lists, other_data_lists):
                """
                Create a scatter plot showing:
                - X-axis: bond amount (£)
                - Y-axis: non-compliance (%)
                - Point color or size: total money paid (non_compliance × bond)

                Assumes bond amount is encoded in the scenario name, e.g., 'bond_800k'.
                """
                import matplotlib.pyplot as plt
                import numpy as np
                import os
                import re

                scatter_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(scatter_folder, exist_ok=True)
                file_path = os.path.join(scatter_folder, "scatter_noncompliance_vs_bond.png")

                bond_vals = []
                noncompliance_vals = []
                total_money_vals = []
                labels = []

                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        scenario_label = getattr(plot_data, 'scenario', f"Scenario {i+1}")
                        print(scenario_label)
                        timesteps = sorted(other_data.keys(), key=int)
                        first_timestep = timesteps[0]
                        nc = other_data[first_timestep]["non_compliance"]

                        # Extract bond amount from name (e.g., "bond_800k" → 800000)
                        match = re.search(r"bond_(\d+)k", scenario_label.lower())
                        if match:
                                bond = int(match.group(1)) * 1_000
                        else:
                                bond = 0  # e.g., for "baseline"

                        total = bond * nc
                        bond_vals.append(bond)
                        noncompliance_vals.append(nc)
                        total_money_vals.append(total)
                        labels.append(scenario_label)

                plt.figure(figsize=(10, 6))
                scatter = plt.scatter(bond_vals, noncompliance_vals, c=total_money_vals, s=100, cmap='viridis', zorder=3)
                plt.colorbar(scatter, label="Total Money Paid (£)")
                
                for x, y, label in zip(bond_vals, noncompliance_vals, labels):
                        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

                plt.xlabel("Bond Amount (£)")
                plt.ylabel("Non-Compliance (%)")
                plt.title("Non-Compliance vs. Bond Level")
                plt.grid(True, zorder=0)
                plt.tight_layout()
                plt.savefig(file_path, dpi=300)
                print(f"Scatter plot saved to {file_path}")
        
        def comparison_scatter_noncompliance_vs_bond(self, plot_data_lists, other_data_lists):
                """
                Create a scatter plot showing:
                - X-axis: bond amount (£)
                - Y-axis: non-compliance (%)
                - Point color or size: total money paid (non_compliance × bond)
                Labels now show total money in millions with a $ sign.
                """
                import matplotlib.pyplot as plt
                import numpy as np
                import os
                import re

                scatter_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(scatter_folder, exist_ok=True)
                file_path = os.path.join(scatter_folder, "scatter_noncompliance_vs_bond.png")

                bond_vals = []
                noncompliance_vals = []
                total_money_vals = []

                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        scenario_label = getattr(plot_data, 'scenario', f"Scenario {i+1}")
                        timesteps = sorted(other_data.keys(), key=int)
                        first_timestep = timesteps[-1]
                        nc = other_data[first_timestep]["non_compliance"]

                        # Extract bond amount from name (e.g., "bond_800k" → 800000)
                        match = re.search(r"bond_(\d+)k", scenario_label.lower())
                        if match:
                                bond = int(match.group(1)) * 1_000
                        else:
                                bond = 0  # e.g., for "baseline"

                        total = bond * nc
                        bond_vals.append(bond)
                        noncompliance_vals.append(nc)
                        total_money_vals.append(total)

                plt.figure(figsize=(10, 6))
                scatter = plt.scatter(bond_vals, noncompliance_vals, c=total_money_vals, s=100, cmap='viridis', zorder=3)
                plt.colorbar(scatter, label="Total Money Paid (£)")

                # Annotate with rounded money values in millions
                for x, y, total in zip(bond_vals, noncompliance_vals, total_money_vals):
                        label = f"${round(total / 1_000_000):,}M"
                        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

                plt.xlabel("Bond Amount (£)")
                plt.ylabel("Count of Derelicts")
                plt.title("Final Year: Derelict Count vs Bond Pot")
                plt.grid(True, zorder=0)
                plt.tight_layout()
                plt.savefig(file_path, dpi=300)
                print(f"Scatter plot saved to {file_path}")

        def comparison_scatter_bond_vs_umpy(self, plot_data_lists, other_data_lists):
                """
                Scatter plot showing:
                - X-axis: bond amount (£)
                - Y-axis: total UMPY (kg) at each timestep
                - Point color: simulation year (from timestep index)
                
                Labels show rounded money paid in millions ($Xm), where:
                total_money = non_compliance × bond
                """
                import matplotlib.pyplot as plt
                import numpy as np
                import os
                import re

                scatter_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(scatter_folder, exist_ok=True)
                file_path = os.path.join(scatter_folder, "scatter_bond_vs_umpy.png")

                bond_vals = []
                umpy_vals = []
                total_money_vals = []
                year_vals = []

                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        scenario_label = getattr(plot_data, 'scenario', f"Scenario {i+1}")

                        # Extract bond amount from scenario name
                        match = re.search(r"bond_(\d+)k", scenario_label.lower())
                        bond = int(match.group(1)) * 1_000 if match else 0

                        timesteps = sorted(other_data.keys(), key=int)

                        for t in timesteps:
                                timestep_data = other_data[t]

                                umpy = np.sum(timestep_data["umpy"])
                                non_compliance = timestep_data["non_compliance"]
                                total_money = bond * non_compliance

                                bond_vals.append(bond)
                                umpy_vals.append(umpy)
                                total_money_vals.append(total_money)
                                year_vals.append(int(t))  # Assuming timestep = simulation year

                # Create the scatter plot
                plt.figure(figsize=(10, 6))
                scatter = plt.scatter(bond_vals, umpy_vals, c=year_vals, cmap='plasma', s=100, zorder=3)
                cbar = plt.colorbar(scatter)
                cbar.set_label("Simulation Year")

                # Annotate each point with $Xm
                for x, y, total in zip(bond_vals, umpy_vals, total_money_vals):
                        label = f"${round(total / 1_000_000):,}M"
                        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)

                plt.xlabel("Bond Amount (£)")
                plt.ylabel("Total UMPY (kg)")
                plt.title("Total UMPY vs. Bond Level by Year")
                plt.grid(True, zorder=0)
                plt.tight_layout()
                plt.savefig(file_path, dpi=300)
                print(f"Scatter plot saved to {file_path}")

        def comparison_umpy_vs_final_metrics(self, plot_data_lists, other_data_lists):
                """
                Create side-by-side scatter plots:
                - Final UMPY vs Total Object Count
                - Final UMPY vs Collision Probability
                - Final UMPY vs Derelict Count (split by naturally compliant vs not)
                """
                import matplotlib.pyplot as plt
                import numpy as np
                from matplotlib.lines import Line2D
                import os
                import re
                import json

                scatter_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(scatter_folder, exist_ok=True)
                file_path = os.path.join(scatter_folder, "umpy_vs_metrics.png")

                umpy_vals = []
                total_counts = []
                collision_probs = []
                derelict_nat_vals = []
                derelict_non_vals = []
                colors_nat = []
                colors_non = []
                markers_nat = []
                markers_non = []
                labels = []

                tax_counter = 1

                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        scenario_label = getattr(plot_data, 'scenario', f"Scenario {i+1}")
                        scenario_folder = scenario_label.lower()

                        timesteps = sorted(other_data.keys(), key=int)
                        final_ts = timesteps[-1]

                        # Final UMPY and collision probability
                        final_umpy = np.sum(other_data[final_ts]["umpy"])
                        final_cp = np.sum(other_data[final_ts].get("collision_probability_all_species", []))

                        # Final object count
                        total = 0
                        for sp, arr in plot_data.data.items():
                                arr_np = np.array(arr)
                                if arr_np.ndim == 2:
                                        total += np.sum(arr_np[-1, :])

                        # Get derelict array
                        derelict_arr = np.array(plot_data.data.get("N_223kg", []))
                        final_derelicts = derelict_arr[-1] if derelict_arr.ndim == 2 else None
                        if final_derelicts is None:
                                continue

                        # Load econ_params JSON
                        scenario_path = os.path.join(self.simulation_folder, scenario_label)
                        econ_file = next((f for f in os.listdir(scenario_path) if f.startswith("econ_params") and f.endswith(".json")), None)
                        if not econ_file:
                                continue
                        with open(os.path.join(scenario_path, econ_file), "r") as f:
                                econ = json.load(f)

                        nat_vec = np.array(econ.get("naturally_compliant_vector", []))
                        if len(nat_vec) != len(final_derelicts):
                                continue

                        nat_count = np.sum(final_derelicts[nat_vec == 1])
                        non_count = np.sum(final_derelicts[nat_vec == 0])

                        # Scenario style
                        is_tax = "tax" in scenario_folder
                        is_25yr = "25yr" in scenario_folder
                        bond_match = re.findall(r'\d+', scenario_folder)
                        bond_value = int(bond_match[0]) if bond_match else None
                        labels.append(f"{bond_value}k" if bond_value else "0k")

                        if is_tax:
                                color = "blue"
                                marker = "s"
                        elif bond_value == 0:
                                color = "orange"
                                marker = "o"
                        elif bond_value is None:
                                color = "green"
                                marker = "o"
                        else:
                                color = "orange" if is_25yr else "green"
                                marker = "o"

                        # Store all values
                        umpy_vals.append(final_umpy)
                        total_counts.append(total)
                        collision_probs.append(final_cp)

                        derelict_nat_vals.append(nat_count)
                        derelict_non_vals.append(non_count)
                        colors_nat.append(color)
                        colors_non.append(color)
                        markers_nat.append(marker)
                        markers_non.append(marker)

                # --- Plot ---
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

                # 1. UMPY vs Object Count
                for x, y, color, marker in zip(umpy_vals, total_counts, colors_nat, markers_nat):
                        ax1.scatter(x, y, color=color, marker=marker, s=90)
                ax1.set_xlabel("Final UMPY (kg/year)", fontsize=14, fontweight="bold")
                ax1.set_ylabel("Final Total Count of Objects", fontsize=14, fontweight="bold")
                ax1.set_title("UMPY vs Object Count", fontsize=14, fontweight="bold")
                ax1.grid(True)
                ax1.tick_params(labelsize=12)

                # 2. UMPY vs Collision Probability
                for x, y, color, marker in zip(umpy_vals, collision_probs, colors_nat, markers_nat):
                        ax2.scatter(x, y, color=color, marker=marker, s=90)
                ax2.set_xlabel("Final UMPY (kg/year)", fontsize=14, fontweight="bold")
                ax2.set_ylabel("Final Collision Probability", fontsize=14, fontweight="bold")
                ax2.set_title("UMPY vs Collision Probability", fontsize=14, fontweight="bold")
                ax2.grid(True)
                ax2.tick_params(labelsize=12)

                # 3. UMPY vs Derelict Counts (Split)
                for x, y, color, marker, bond in zip(umpy_vals, derelict_nat_vals, colors_nat, markers_nat, labels):
                        ax3.scatter(x, y, color='green', marker=marker, s=90)
                        ax3.annotate(f"B = {bond}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)

                for x, y, color, marker, bond in zip(umpy_vals, derelict_non_vals, colors_non, markers_non, labels):
                        ax3.scatter(x, y, color='red', marker=marker, s=90)
                        ax3.annotate(f"B = {bond}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)

                ax3.set_xlabel("Final UMPY (kg/year)", fontsize=14, fontweight="bold")
                ax3.set_ylabel("Derelict Count (N_223kg)", fontsize=14, fontweight="bold")
                ax3.set_title("UMPY vs Derelicts (Split)", fontsize=14, fontweight="bold")
                ax3.grid(True)
                ax3.tick_params(labelsize=12)

                # --- Shared Legend ---
                legend_elements = [
                        Line2D([0], [0], marker='o', color='w', label='5yr PMD', markerfacecolor='green', markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='25yr PMD', markerfacecolor='orange', markersize=10),
                        Line2D([0], [0], marker='s', color='w', label='Tax Scenario', markerfacecolor='blue', markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='Nat. Compliant Derelicts', markerfacecolor='green', markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='Non-Compliant Derelicts', markerfacecolor='red', markersize=10)
                ]
                ax3.legend(handles=legend_elements, loc='upper left', fontsize=11, title="Scenario Type")

                plt.tight_layout()
                plt.savefig(file_path, dpi=300)
                print(f"Comparison plots saved to {file_path}")

        def comparison_total_welfare_vs_time(self, plot_data_lists, other_data_lists):
                """
                Plot total welfare over time for each scenario.
                - Each line corresponds to one scenario.
                - Welfare is summed across all S-prefixed species per timestep.
                - Welfare = 100 × (sum of satellites)^2 at each timestep.
                """
                import numpy as np
                import matplotlib.pyplot as plt
                import os

                coef = 1e2
                plt.figure(figsize=(10, 6))

                for plot_data in plot_data_lists:
                        species_data = {sp: np.array(data) for sp, data in plot_data.data.items()}
                        s_species_names = [sp for sp in species_data if sp.startswith("S")]

                        if not s_species_names:
                                print(f"No S-prefixed species found in scenario '{plot_data.scenario}'")
                                continue

                        # Sum all S-prefixed species into one welfare curve
                        total_sats = None
                        for sp in s_species_names:
                                arr = species_data[sp]  # (timesteps, shells)
                                sats = np.sum(arr, axis=1)  # (timesteps,)
                                total_sats = sats if total_sats is None else total_sats + sats

                        welfare = coef * (total_sats ** 2)
                        label = getattr(plot_data, 'scenario', 'Unnamed Scenario')
                        plt.plot(welfare, label=label, linewidth=2)

                plt.title("Total Welfare Over Time by Scenario", fontsize=14, fontweight='bold')
                plt.xlabel("Year", fontsize=12)
                plt.ylabel("Welfare (Summed Across S-Prefixed Species)", fontsize=12)
                plt.legend(title="Scenario", fontsize=10)
                plt.grid(True)
                plt.tight_layout()

                # Save
                outdir = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(outdir, exist_ok=True)
                file_path = os.path.join(outdir, "total_welfare_across_scenarios.png")
                plt.savefig(file_path, dpi=300)

                print(f"Scenario-wise total welfare plot saved to {file_path}")


        def comparison_object_counts_vs_bond(self, plot_data_lists, other_data_lists):
                """
                Compare number of derelicts (N_223kg) and fringe satellites (Su) across 5yr and 25yr PMD scenarios.
                Separates naturally compliant vs non-compliant derelicts using econ_params.

                X-axis: Bond amount ($k)
                Y-axis: Number of objects
                """
                import os
                import json
                import re
                import numpy as np
                import matplotlib.pyplot as plt

                root_folder = self.simulation_folder

                # Separate derelicts by compliance category
                bond_5yr_nat, nat_5yr = [], []
                bond_5yr_non, non_5yr = [], []
                bond_25yr_nat, nat_25yr = [], []
                bond_25yr_non, non_25yr = [], []

                # Fringe satellite counts
                bond_5yr_Su, Su_5yr = [], []
                bond_25yr_Su, Su_25yr = [], []

                for folder_name in os.listdir(root_folder):
                        folder_path = os.path.join(root_folder, folder_name)
                        if not os.path.isdir(folder_path):
                                continue

                        is_25yr = folder_name.endswith("25yr")
                        match = re.findall(r"\d+", folder_name)
                        bond = float(match[0]) if match else None
                        if bond is None:
                                continue

                        species_file = next((f for f in os.listdir(folder_path) if f.startswith("species_data")), None)
                        econ_file = next((f for f in os.listdir(folder_path) if f.startswith("econ_params") and f.endswith(".json")), None)
                        if not econ_file:
                                continue

                        with open(os.path.join(folder_path, econ_file), "r") as f:
                                econ = json.load(f)
                        if not species_file or not econ_file:
                                continue

                        with open(os.path.join(folder_path, species_file), "r") as f:
                                data = json.load(f)
                        with open(os.path.join(folder_path, econ_file), "r") as f:
                                econ = json.load(f)

                        try:
                                N_arr = np.array(data["N_223kg"])  # (timesteps, shells)
                                Su_arr = np.array(data["Su"])      # (timesteps, shells)
                                final_N = N_arr[-1]
                                final_Su = np.sum(Su_arr[-1])
                        except (KeyError, IndexError, TypeError):
                                continue

                        nat_vec = np.array(econ.get("naturally_compliant_vector", []))
                        if len(nat_vec) != len(final_N):
                                continue  # mismatch in dimensions

                        nat_sum = np.sum(final_N[nat_vec == 1])
                        non_sum = np.sum(final_N[nat_vec == 0])

                        if is_25yr:
                                bond_25yr_nat.append(bond)
                                nat_25yr.append(nat_sum)
                                bond_25yr_non.append(bond)
                                non_25yr.append(non_sum)
                                bond_25yr_Su.append(bond)
                                Su_25yr.append(final_Su)
                        else:
                                bond_5yr_nat.append(bond)
                                nat_5yr.append(nat_sum)
                                bond_5yr_non.append(bond)
                                non_5yr.append(non_sum)
                                bond_5yr_Su.append(bond)
                                Su_5yr.append(final_Su)

                # --- Sort helper ---
                def sort_by_bond(bonds, vals):
                        zipped = sorted(zip(bonds, vals), key=lambda x: x[0])
                        return zip(*zipped) if zipped else ([], [])

                bond_5yr_nat, nat_5yr = sort_by_bond(bond_5yr_nat, nat_5yr)
                bond_5yr_non, non_5yr = sort_by_bond(bond_5yr_non, non_5yr)
                bond_25yr_nat, nat_25yr = sort_by_bond(bond_25yr_nat, nat_25yr)
                bond_25yr_non, non_25yr = sort_by_bond(bond_25yr_non, non_25yr)
                bond_5yr_Su, Su_5yr = sort_by_bond(bond_5yr_Su, Su_5yr)
                bond_25yr_Su, Su_25yr = sort_by_bond(bond_25yr_Su, Su_25yr)

                # --- Plot ---
                plt.figure(figsize=(10, 6))

                # Derelicts
                plt.scatter(bond_5yr_nat, nat_5yr, marker='o', color='green', label='5yr PMD - Nat. Compliant Derelicts')
                plt.scatter(bond_5yr_non, non_5yr, marker='o', color='red', label='5yr PMD - Non-Comp. Derelicts')
                plt.scatter(bond_25yr_nat, nat_25yr, marker='^', color='green', label='25yr PMD - Nat. Compliant Derelicts')
                plt.scatter(bond_25yr_non, non_25yr, marker='^', color='red', label='25yr PMD - Non-Comp. Derelicts')

                # Fringe satellites
                plt.plot(bond_5yr_Su, Su_5yr, marker='s', linestyle='-', color='blue', label='5yr PMD - Fringe Sats')
                plt.plot(bond_25yr_Su, Su_25yr, marker='s', linestyle='--', color='blue', label='25yr PMD - Fringe Sats')

                plt.xlabel("Lifetime Bond Amount, $ (k)", fontsize=14, fontweight='bold')
                plt.ylabel("Number of Objects", fontsize=14, fontweight='bold')
                plt.xticks(fontsize=12, fontweight='bold')
                plt.yticks(fontsize=12, fontweight='bold')
                plt.grid(True)
                plt.legend(fontsize=12, loc='upper right')
                plt.tight_layout()

                for spine in plt.gca().spines.values():
                        spine.set_linewidth(1.5)

                # Save and show
                file_path = os.path.join(self.simulation_folder, "comparisons", "object_counts_vs_bond_split.png")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                plt.savefig(file_path, dpi=300)
                print(f"Split object count plot saved to {file_path}")



        def UMPY(self, plot_data, other_data):
                 # 1) Sort the timesteps and prepare arrays
                timesteps = sorted(other_data.keys(), key=int)  
                umpy_sums = []

                # 2) For each timestep, sum the 'umpy' list
                for ts in timesteps:
                        umpy_list = other_data[ts]["umpy"]  # This is assumed to be a list of floats
                        total_umpy = np.sum(umpy_list)      # Sum across all shells
                        umpy_sums.append(total_umpy)

                # 3) Create the plot
                plt.figure(figsize=(8, 5))
                plt.plot(timesteps, umpy_sums, marker='o', label="Total UMPY (kg/year)")

                # 4) Labels and title
                plt.xlabel("Year (timestep)")
                plt.ylabel("UMPY (kg/year)")
                plt.title("UMPY Evolution Over Time")
                plt.legend()
                plt.tight_layout()

                # 5) Save the figure
                save_path = os.path.join(plot_data.path, "umpy_time_evolution.png")
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()


        def count_by_shell_and_time_per_species(self, plot_data, other_data):
                species_data = {sp: np.array(data) for sp, data in plot_data.data.items()}

                # Ensure folder exists
                os.makedirs(plot_data.path, exist_ok=True)

                for sp, data in species_data.items():
                        plt.figure(figsize=(8, 6))
                        plt.imshow(data.T, aspect='auto', cmap='viridis', origin='lower')
                        plt.colorbar(label='Value')
                        plt.title(f'Heatmap for Species {sp}')
                        plt.xlabel('Year')
                        plt.ylabel('Shell Mid Altitude (km)')
                        plt.xticks(ticks=range(data.shape[0]), labels=range(1, data.shape[0] + 1))
                        plt.yticks(ticks=range(data.shape[1]), labels=self.HMid)

                        # Save the plot to the designated folder
                        file_path = os.path.join(plot_data.path, f"count_over_time_{sp}.png")
                        plt.savefig(file_path, dpi=300, bbox_inches='tight')
                        plt.close()

        def combined_count_by_shell_and_time(self, plot_data, other_data):
                """
                Generate and save a single combined heatmap for all species.
                """
                species_data = {sp: np.array(data) for sp, data in plot_data.data.items()}

                # Calculate number of rows and columns for the subplot grid
                num_species = len(species_data)
                cols = int(np.ceil(np.sqrt(num_species)))
                rows = int(np.ceil(num_species / cols))

                # Create the figure
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6), constrained_layout=True)

                # Flatten axes for easy iteration (handles edge cases where rows * cols > num_species)
                axes = axes.flatten() if num_species > 1 else [axes]

                for ax, (sp, data) in zip(axes, species_data.items()):
                        im = ax.imshow(data.T, aspect='auto', cmap='viridis', origin='lower')
                        ax.set_title(f'Species {sp}')
                        ax.set_xlabel('Year')
                        ax.set_ylabel('Shell Mid Altitude (km)')
                        ax.set_xticks(range(data.shape[0]))
                        ax.set_xticklabels(range(1, data.shape[0] + 1))
                        ax.set_yticks(range(data.shape[1]))
                        ax.set_yticklabels(self.HMid)
                        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

                # Turn off unused subplots
                for ax in axes[len(species_data):]:
                        ax.axis('off')

                # Save the combined plot
                combined_file_path = os.path.join(plot_data.path, "combined_species_heatmaps.png")
                plt.savefig(combined_file_path, dpi=300, bbox_inches='tight')
                plt.close()


        ## These plots dont work
        # def ror_cp_and_launch_rate(self, plot_data, other_data):
        #         """
        #         Generate and save a combined plot for time evolution of different parameters (RoR, Collision Probability, Launch Rate).
        #         """
        #         # Extract keys (timesteps) and sort
        #         timesteps = sorted(other_data.keys(), key=int)

        #         # Get number of altitude shells (assuming all timesteps have the same length)
        #         num_altitude_shells = len(other_data[timesteps[0]]["ror"])

        #         # Prepare the figure
        #         fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        #         # Color map for time evolution
        #         colors = cm.viridis(np.linspace(0, 1, len(timesteps)))

        #         # Static Plot
        #         for idx, timestep in enumerate(timesteps):
        #                 ror = other_data[timestep]["ror"]
        #                 collision_prob = other_data[timestep]["collision_probability"]
        #                 launch_rate = other_data[timestep]["launch_rate"]
        #                 excess_returns = other_data[timestep]["excess_returns"]

        #                 # Number of species is inferred from array length and known number of shells
        #                 num_species = len(ror) // self.n_shells

        #                 for species_idx in range(num_species):
        #                         label = f"Year {timestep} - Species {species_idx + 1}"
        #                         start = species_idx * self.n_shells
        #                         end = (species_idx + 1) * self.n_shells

        #                         ror_slice = ror[start:end]
        #                         cp_slice = collision_prob[start:end]
        #                         lr_slice = launch_rate[start:end]
        #                         er_slice = excess_returns[start:end]

        #                         if len(ror_slice) == self.n_shells:
        #                                 axes[0].plot(self.HMid, ror_slice, color=colors[idx], label=label)
        #                                 axes[1].plot(self.HMid, cp_slice, color=colors[idx])
        #                                 axes[2].plot(self.HMid, lr_slice, color=colors[idx])
        #                                 axes[3].plot(self.HMid, er_slice, color=colors[idx])
        #                         else:
        #                                 print(f"Warning: Skipping timestep {timestep}, species {species_idx} due to mismatched shell size.")


        #         axes[0].set_title("Rate of Return (RoR)")
        #         axes[1].set_title("Collision Probability")
        #         axes[2].set_title("Launch Rate")

        #         # Set labels and ticks
        #         for ax in axes:
        #                 ax.set_xlabel("Shell - Mid Altitude (km)")
        #                 ax.set_ylabel("Value")
        #                 ax.set_xticklabels(self.HMid)

        #         # Add a legend
        #         fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)

        #         # Tight layout
        #         plt.tight_layout()

        #         # Save the combined plot
        #         combined_file_path = os.path.join(plot_data.path, "combined_time_evolution.png")
        #         plt.savefig(combined_file_path, dpi=300, bbox_inches='tight')
        #         plt.close()

        # def ror_cp_and_launch_rate_gif(self, plot_data, other_data):
        #         """
        #         Generate and save an animated plot for the time evolution of different parameters (RoR, Collision Probability, Launch Rate).
        #         """
        #         # Extract keys (timesteps) and sort
        #         timesteps = sorted(other_data.keys(), key=int)

        #         # Get number of altitude shells (assuming all timesteps have the same length)
        #         num_altitude_shells = len(other_data[timesteps[0]]["ror"])

        #         # Determine global min/max for each metric across all timesteps
        #         ror_values = [val for timestep in timesteps for val in other_data[timestep]["ror"]]
        #         collision_values = [val for timestep in timesteps for val in other_data[timestep]["collision_probability"]]
        #         launch_values = [val for timestep in timesteps for val in other_data[timestep]["launch_rate"]]

        #         ror_min, ror_max = min(ror_values), max(ror_values)
        #         collision_min, collision_max = min(collision_values), max(collision_values)
        #         launch_min, launch_max = min(launch_values), max(launch_values)

        #         # Create the figure and axes
        #         fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        #         def update(frame):
        #                 timestep = timesteps[frame]
        #                 ror = other_data[timestep]["ror"]
        #                 collision_prob = other_data[timestep]["collision_probability"]
        #                 launch_rate = other_data[timestep]["launch_rate"]

        #                 for ax in axes:
        #                         ax.clear()

        #                         # Plot each metric with fixed y-axis limits
        #                         axes[0].plot(range(num_altitude_shells), ror, color='b')
        #                         axes[0].set_ylim(ror_min, ror_max)
        #                         axes[0].set_title(f"Rate of Return (RoR) - Year {timestep}")
        #                         axes[0].set_xlabel("Shell - Mid Altitude (km)")
        #                         axes[0].set_ylabel("RoR")
        #                         axes[0].set_xticks(range(len(self.HMid)))  # Ensure correct number of ticks
        #                         axes[0].set_xticklabels(self.HMid)

        #                         axes[1].plot(range(num_altitude_shells), collision_prob, color='r')
        #                         axes[1].set_ylim(collision_min, collision_max)
        #                         axes[1].set_title(f"Collision Probability - Year {timestep}")
        #                         axes[1].set_xlabel("Shell - Mid Altitude (km)")
        #                         axes[1].set_ylabel("Collision Probability")
        #                         axes[1].set_xticks(range(len(self.HMid)))  # Ensure correct number of ticks
        #                         axes[1].set_xticklabels(self.HMid)

        #                         axes[2].plot(range(num_altitude_shells), launch_rate, color='g')
        #                         axes[2].set_ylim(launch_min, launch_max)
        #                         axes[2].set_title(f"Launch Rate - Year {timestep}")
        #                         axes[2].set_xlabel("Shell - Mid Altitude (km)")
        #                         axes[2].set_ylabel("Launch Rate")
        #                         axes[2].set_xticks(range(len(self.HMid)))  # Ensure correct number of ticks
        #                         axes[2].set_xticklabels(self.HMid)

        #                 plt.tight_layout()
                
        #         # Create the animation
        #         ani = animation.FuncAnimation(fig, update, frames=len(timesteps), repeat=True)

        #         # Save as GIF
        #         combined_file_path = os.path.join(plot_data.path, "space_metrics_evolution.gif")
        #         ani.save(combined_file_path, writer="pillow", fps=2)

        #         plt.close()