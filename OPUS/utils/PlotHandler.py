import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import math
import pickle
import re

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
                
                # Check if data was successfully loaded
                if self.data is None:
                        raise ValueError(f"Could not load data for scenario {scenario} from {path}")
                
                self.species_names = MOCAT.scenario_properties.species_names
                self.n_shells = MOCAT.scenario_properties.n_shells
                self.n_species = MOCAT.scenario_properties.species_length
                self.HMid = MOCAT.scenario_properties.HMid

                # Get derelict species names from the species configuration
                try:
                        self.derelict_species_names = MOCAT.scenario_properties.pmd_debris_names
                except AttributeError:
                        # Fallback: get derelict species from species names that start with 'N' or 'B'
                        self.derelict_species_names = [name for name in MOCAT.scenario_properties.species_names 
                                                      if name.startswith('N') or name.startswith('B')]


        def get_other_data(self):
                return self.other_data

        def load_data(self, path):
                """
                Load the data from the scenario folder, prioritizing files with 'species_data' and 'other_results'
                """
                if not os.path.exists(path):
                        print(f"Error: {path} does not exist.")
                        return None, None, None
                
                # Find the JSON files in the folder
                json_files = [f for f in os.listdir(path) if f.endswith(".json")]

                if len(json_files) == 0:
                        print(f"Error: No JSON file found in {path}.")
                        return None, None, None
                
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
                
                # Convert new data structure to expected format if needed
                if data is not None:
                        data = self._convert_data_structure(data)
                
                return data, other_data, econ_params

        def _convert_data_structure(self, data):
                """
                Convert the new nested data structure (species -> year -> array) 
                to the expected format (species -> 2D array with time and shells)
                """
                converted_data = {}
                
                for species_name, species_data in data.items():
                        if isinstance(species_data, dict):
                                # New structure: species -> year -> array
                                # Convert to: species -> 2D array (time, shells)
                                years = sorted(species_data.keys(), key=int)
                                arrays = []
                                
                                for year in years:
                                        year_data = species_data[year]
                                        if isinstance(year_data, list):
                                                arrays.append(year_data)
                                
                                if arrays:
                                        # Stack arrays to create 2D array (time, shells)
                                        converted_data[species_name] = np.array(arrays)
                                else:
                                        converted_data[species_name] = species_data
                        else:
                                # Already in expected format
                                converted_data[species_name] = species_data
                
                return converted_data


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
                try:
                        self.HMid = self.MOCAT.scenario_properties.HMid
                        self.n_shells = self.MOCAT.scenario_properties.n_shells
                except:
                        self.HMid = self.MOCAT.HMid
                        self.n_shells = self.MOCAT.n_shells

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
                                # create the folder
                                os.makedirs(scenario_folder, exist_ok=True)
                                # print(f"Error: {scenario_folder} folder does not exist. Skipping scenario...")
                                continue
                        else: 
                                print("Generating plots for scenario: ", scenario)

                                # Build a PlotData object and then pass to the plotting functions
                                try:
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
                                except ValueError as e:
                                        print(f"Warning: Skipping scenario {scenario} - {e}")
                                        continue

                if comparison:
                        self._comparison_plots(plot_data_list, other_data_list)
                
        def _comparison_plots(self, plot_data_lists, other_data_lists):
                """
                Run all plot functions that start with 'comparison_', ignoring others.
                """
                for attr_name in dir(self):
                        # Grab the attribute; see if it's a callable (method)
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
                                # Updated block to handle nested econ_params
                                elif attr_name.startswith("econ_"):
                                        print(f"Creating economic plots for {attr_name}...")
                                        plot_method = getattr(self, attr_name)
                                        # Loop through each species in the econ_params dict
                                        for species_name, species_econ_data in econ_params.items():
                                                print(f"  ... for species: {species_name}")
                                                # Create a species-specific path
                                                species_plot_path = os.path.join(plot_data.path, species_name)
                                                # Pass the inner dictionary (species_econ_data)
                                                plot_method(species_plot_path, species_econ_data)
                
                # Create 3D plots for maneuvers and collisions
                # self._create_3d_maneuver_plots(plot_data, other_data)
                # self._create_3d_collision_plots(plot_data, other_data)
                
                # Create economic metrics plots
                self._create_economic_metrics_plots(plot_data, other_data, econ_params)


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

                        if idx == 0:  # First plot
                                ax.set_title("LEO Species Total")
                        else:
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
                # plt.title("UMPY Evolution Over Time (All Scenarios)")  # Removed overall title
                plt.legend()
                plt.tight_layout()

                # 5) Save the figure using the first plot_data's path 
                out_path = os.path.join(comparison_folder, "umpy_over_time.png")
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Comparison UMPY plot saved to {out_path}")

        # Removed the first (duplicate) definition of comparison_scatter_noncompliance_vs_bond
        
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
                        
                        # Sum all non-compliance values if it's a dictionary
                        if isinstance(nc, dict):
                                nc_total = sum(nc.values())
                        else:
                                nc_total = nc

                        # Extract bond amount from name (e.g., "bond_800k" → 800000)
                        match = re.search(r"bond_(\d+)k", scenario_label.lower())
                        if match:
                                bond = int(match.group(1)) * 1_000
                        else:
                                bond = 0  # e.g., for "baseline"

                        total = bond * nc_total
                        bond_vals.append(bond)
                        noncompliance_vals.append(nc_total)
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


        def comparison_species_and_derelicts_response_to_bonds(self, plot_data_lists, other_data_lists):
                """
                Generate a 2x3 comparison figure that shows how counts respond to bond levels
                for two PMD lifetimes (5yr solid, 25yr dashed):
                  - Top row: S, Su, Sns totals (final year)
                  - Bottom row: selected derelict mass classes (final year)

                Notes
                - Derelicts are pulled from N_* species present in the data. We try to
                  select the closest available mass buckets to 521kg, 700kg and 20kg.
                - Uses the linked-species compliant split infrastructure already present
                  elsewhere in this module to access scenario data; here we only need final
                  totals per species at the final timestep.
                - Scenarios are expected to include pattern "bond_{K}k_{Y}yr" in their
                  names to infer the bond and the PMD lifetime.
                """
                import os
                import re
                import numpy as np
                import matplotlib.pyplot as plt

                comparison_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(comparison_folder, exist_ok=True)

                # Targets for derelict buckets (kg). We'll choose the closest available N_{mass}kg
                target_der_mass = [521.0, 700.0, 20.0]

                # Helpers ----------------------------------------------------
                def parse_der_mass(name: str):
                        m = re.match(r"^N_([0-9]+(?:\.[0-9]+)?)kg", name)
                        return float(m.group(1)) if m else None

                def sum_species_final(plot_data, selector) -> float:
                        total = 0.0
                        for sp_name, sp_data in plot_data.data.items():
                                if selector(sp_name):
                                        arr = np.array(sp_data)
                                        if arr.ndim == 2 and arr.shape[0] > 0:
                                                total += float(np.sum(arr[-1, :]))
                        return total

                # First pass: discover available derelict masses across scenarios
                available_masses = set()
                for plot_data in plot_data_lists:
                        for sp_name in plot_data.data.keys():
                                if sp_name.startswith('N_'):
                                        mass = parse_der_mass(sp_name)
                                        if mass is not None:
                                                available_masses.add(mass)
                available_masses = sorted(list(available_masses))

                # Map each target (521, 700, 20) to the closest available mass
                chosen_der_masses = {}
                for target in target_der_mass:
                        if not available_masses:
                                continue
                        idx = int(np.argmin(np.abs(np.array(available_masses) - target)))
                        chosen_der_masses[target] = available_masses[idx]

                # Data buckets by lifetime
                series_keys = ['S', 'Su', 'Sns']
                # Create keys like 'N_521kg', etc., based on chosen masses
                der_keys = []
                for target in target_der_mass:
                        if target in chosen_der_masses:
                                chosen = chosen_der_masses[target]
                                der_keys.append(f"N_{int(chosen) if chosen.is_integer() else chosen}kg")

                data_5 = {k: {'bond': [], 'y': []} for k in series_keys + der_keys}
                data_25 = {k: {'bond': [], 'y': []} for k in series_keys + der_keys}

                # Collect ----------------------------------------------------
                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        scenario_label = getattr(plot_data, 'scenario', f"Scenario {i+1}")
                        scenario_lower = scenario_label.lower()
                        
                        # Explicitly check for Baseline scenario
                        if 'baseline' in scenario_lower:
                                bond_k = 0
                                lifetime = 5  # Baseline is 0 bond, 5 year PMD
                        else:
                                m = re.search(r"bond_(\d+)k_(\d+)yr", scenario_lower)
                                if not m:
                                        # Treat as baseline ($0 bond). Infer lifetime from name if present, default 5yr
                                        bond_k = 0
                                        m_life = re.search(r"(?:^|_)(5|25)yr", scenario_lower)
                                        lifetime = int(m_life.group(1)) if m_life else 5
                                else:
                                        bond_k = int(m.group(1))
                                        lifetime = int(m.group(2))

                        # Final timestep
                        timesteps = sorted(other_data.keys(), key=int)
                        if not timesteps:
                                continue
                        # Species totals (final year)
                        total_S   = sum_species_final(plot_data, lambda n: n.startswith('S') and not n.startswith('Su') and not n.startswith('Sns'))
                        total_Su  = sum_species_final(plot_data, lambda n: n == 'Su')
                        total_Sns = sum_species_final(plot_data, lambda n: n == 'Sns')

                        # Derelict buckets by chosen closest mass
                        der_totals = {}
                        for target in target_der_mass:
                                if target not in chosen_der_masses:
                                        continue
                                chosen = chosen_der_masses[target]
                                # species name we will look for (exact match if exists)
                                sp_key_exact = f"N_{int(chosen) if float(chosen).is_integer() else chosen}kg"
                                total = sum_species_final(plot_data, lambda n, key=sp_key_exact: n == key)
                                # If exact not found, sum all derelicts whose parsed mass is within 1 kg of the chosen
                                if total == 0.0:
                                        total = sum_species_final(plot_data, lambda n, ch=chosen: (n.startswith('N_') and (parse_der_mass(n) is not None) and abs(parse_der_mass(n) - ch) <= 1.0))
                                der_totals[sp_key_exact] = total

                        bucket = data_5 if lifetime == 5 else data_25
                        # Add S, Su, Sns
                        bucket['S']['bond'].append(bond_k);   bucket['S']['y'].append(total_S)
                        bucket['Su']['bond'].append(bond_k);  bucket['Su']['y'].append(total_Su)
                        bucket['Sns']['bond'].append(bond_k); bucket['Sns']['y'].append(total_Sns)
                        # Add derelicts
                        for k, v in der_totals.items():
                                if k not in bucket:
                                        bucket[k] = {'bond': [], 'y': []}
                                bucket[k]['bond'].append(bond_k)
                                bucket[k]['y'].append(v)

                # Sort by bond for each series
                def sort_series(dct):
                        for k, obj in dct.items():
                                if obj.get('bond'):
                                        order = np.argsort(obj['bond'])
                                        obj['bond'] = [obj['bond'][idx] for idx in order]
                                        obj['y']    = [obj['y'][idx] for idx in order]
                        return dct

                data_5 = sort_series(data_5)
                data_25 = sort_series(data_25)

                # Figure -----------------------------------------------------
                fig, axes = plt.subplots(2, 3, figsize=(16, 10))
                panels = [
                        ('S',   'S Species Response to Bonds', 0, 0),
                        ('Su',  'Su Species Response to Bonds', 0, 1),
                        ('Sns', 'Sns Species Response to Bonds',0, 2),
                ]
                # Fill bottom row with chosen derelict keys; pad with blanks if <3
                der_for_panels = der_keys[:3] if len(der_keys) >= 3 else der_keys + ['']*(3-len(der_keys))
                for j, dk in enumerate(der_for_panels):
                        title = f"{dk} Derelicts Response to Bonds" if dk else ""
                        panels.append((dk, title, 1, j))

                # Plot helper
                def plot_series(ax, key, title):
                        if not key:
                                ax.axis('off'); return
                        # 5-year
                        if key in data_5 and data_5[key]['bond']:
                                ax.plot(data_5[key]['bond'], data_5[key]['y'], color='C0', marker='o', linestyle='-', label='5-year PMD')
                                for x, y in zip(data_5[key]['bond'], data_5[key]['y']):
                                        ax.text(x, y, f"{y:,.0f}", fontsize=8, va='bottom', ha='center')
                        # 25-year
                        if key in data_25 and data_25[key]['bond']:
                                ax.plot(data_25[key]['bond'], data_25[key]['y'], color='C0', marker='o', linestyle='--', alpha=0.8, label='25-year PMD')
                        ax.set_title(title)
                        ax.set_xlabel('Bond Amount ($)')
                        ax.set_ylabel(f"{key} Species Count")
                        ax.grid(True, alpha=0.3)
                        ax.legend(loc='best', fontsize=8)

                # Draw panels
                for key, title, r, c in panels:
                        plot_series(axes[r][c], key, title)

                plt.tight_layout()
                out_path = os.path.join(comparison_folder, 'species_and_derelicts_response_to_bonds.png')
                plt.savefig(out_path, dpi=300)
                plt.close()
                print(f"Comparison plot saved to {out_path}")

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
                                # Sum all non-compliance values if it's a dictionary
                                if isinstance(non_compliance, dict):
                                        non_compliance_total = sum(non_compliance.values())
                                else:
                                        non_compliance_total = non_compliance
                                total_money = bond * non_compliance_total

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

                # Create comparison folder
                comparison_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(comparison_folder, exist_ok=True)
                file_path = os.path.join(comparison_folder, "umpy_vs_metrics.png")

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

                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        scenario_label = getattr(plot_data, 'scenario', f"Scenario {i+1}")
                        scenario_folder = scenario_label.lower()

                        timesteps = sorted(other_data.keys(), key=int)
                        final_ts = timesteps[-1]

                        # Final UMPY and collision probability
                        umpy_data = other_data[final_ts].get("umpy", [])
                        if isinstance(umpy_data, dict):
                                final_umpy = np.sum(list(umpy_data.values()))
                        else:
                                final_umpy = np.sum(umpy_data)
                        
                        cp_data = other_data[final_ts].get("collision_probability_all_species", [])
                        if isinstance(cp_data, dict):
                                final_cp = np.sum(list(cp_data.values()))
                        else:
                                final_cp = np.sum(cp_data)

                        # Final object count across all species
                        total = 0
                        for sp, species_data in plot_data.data.items():
                                if isinstance(species_data, np.ndarray):
                                        # Data is already converted to numpy array (time, shells)
                                        if species_data.ndim == 2:
                                                total += np.sum(species_data[-1, :])
                                        else:
                                                total += np.sum(species_data)
                                elif isinstance(species_data, dict):
                                        # Data is stored as {year: [shell1, shell2, ...], ...}
                                        year_keys = sorted(species_data.keys(), key=int)
                                        final_year = year_keys[-1]
                                        final_year_data = species_data[final_year]
                                        if isinstance(final_year_data, list):
                                                total += np.sum(final_year_data)
                                elif isinstance(species_data, list) and len(species_data) > 0:
                                        arr_np = np.array(species_data)
                                        if arr_np.ndim == 2:
                                                total += np.sum(arr_np[-1, :])
                                        else:
                                                total += np.sum(arr_np)

                        # Find derelict species (N_* or B_*)
                        derelict_species = [sp for sp in plot_data.data.keys() if sp.startswith('N_') or sp.startswith('B_')]
                        
                        # Calculate total derelicts
                        total_derelicts = 0
                        for derelict_sp in derelict_species:
                                if derelict_sp in plot_data.data:
                                        derelict_data = plot_data.data[derelict_sp]
                                        if isinstance(derelict_data, np.ndarray):
                                                # Data is already converted to numpy array (time, shells)
                                                if derelict_data.ndim == 2:
                                                        total_derelicts += np.sum(derelict_data[-1, :])
                                                else:
                                                        total_derelicts += np.sum(derelict_data)
                                        elif isinstance(derelict_data, dict):
                                                # Data is stored as {year: [shell1, shell2, ...], ...}
                                                year_keys = sorted(derelict_data.keys(), key=int)
                                                final_year = year_keys[-1]
                                                final_year_data = derelict_data[final_year]
                                                if isinstance(final_year_data, list):
                                                        total_derelicts += np.sum(final_year_data)
                                        elif isinstance(derelict_data, list) and len(derelict_data) > 0:
                                                derelict_arr = np.array(derelict_data)
                                                if derelict_arr.ndim == 2:
                                                        total_derelicts += np.sum(derelict_arr[-1, :])
                                                else:
                                                        total_derelicts += np.sum(derelict_arr)

                        # For now, split derelicts 50/50 since we don't have econ_params in memory
                        # This is a simplified approach - in a real implementation you'd want to pass econ_params
                        nat_count = total_derelicts * 0.5  # Simplified split
                        non_count = total_derelicts * 0.5

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

                # Check if we have data to plot
                if not umpy_vals:
                        print("Warning: No data found for umpy_vs_metrics plot")
                        return

                # --- Plot ---
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

                # 1. UMPY vs Object Count
                for x, y, color, marker in zip(umpy_vals, total_counts, colors_nat, markers_nat):
                        ax1.scatter(x, y, color=color, marker=marker, s=90)
                ax1.set_xlabel("Final UMPY (kg/year)", fontsize=14, fontweight="bold")
                ax1.set_ylabel("Final Total Count of Objects", fontsize=14, fontweight="bold")
                ax1.set_title("UMPY vs Total Objects", fontsize=14, fontweight="bold")
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
                ax3.set_ylabel("Derelict Count", fontsize=14, fontweight="bold")
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
                ax3.legend(handles=legend_elements, loc='upper right', fontsize=11, title="Scenario Type")

                plt.tight_layout()
                plt.savefig(file_path, dpi=300)
                plt.close()
                print(f"UMPY vs metrics plot saved to {file_path}")

        def comparison_total_welfare_vs_time(self, plot_data_lists, other_data_lists):
                """
                Plot total welfare over time for each scenario.
                - Plots welfare from satellites: SUM( coef_i * (sum of S-species_i)^2 )
                - Plots total welfare: (welfare from satellites) + (bond revenue)
                """
                import numpy as np
                import matplotlib.pyplot as plt
                import os

                plt.figure(figsize=(10, 6))

                for plot_data, other_data in zip(plot_data_lists, other_data_lists):
                        species_data = {sp: np.array(data) for sp, data in plot_data.data.items()}
                        s_species_names = [sp for sp in species_data.keys() if sp.startswith("S")] 

                        if not s_species_names:
                                print(f"No S-prefixed species found in scenario '{plot_data.scenario}'")
                                continue

                        # --- 1. Calculate base welfare (Welfare WITHOUT revenue) ---
                        
                        econ_params_dict = plot_data.econ_params
                        
                        first_sp_data = next(iter(species_data.values()))
                        welfare_base = np.zeros(first_sp_data.shape[0]) # Shape: (timesteps,)
                        
                        for sp in s_species_names:
                                if sp not in econ_params_dict:
                                        print(f"Warning: No econ_params found for {sp} in {plot_data.scenario}. Skipping its welfare.")
                                        continue
                                
                                species_coef = econ_params_dict[sp].get("coef", 1.0e2)
                                sats_i = np.sum(species_data[sp], axis=1) # Shape: (timesteps,)
                                welfare_base += species_coef * (sats_i ** 2)

                        # --- 2. Calculate bond revenue over time ---
                        
                        bond_amount = 0.0
                        if s_species_names and s_species_names[0] in econ_params_dict:
                                bond_from_json = econ_params_dict[s_species_names[0]].get("bond") 
                                bond_amount = bond_from_json or 0.0
                        
                        print(f"\n--- Plot Debug: {plot_data.scenario} ---")
                        print(f"Using bond_amount: {bond_amount} for calculation.") 
                        
                        timesteps_other = sorted(other_data.keys(), key=int)
                        
                        bond_revenues = [0.0] # Start at year 0 with 0 revenue
                        
                        for ts in timesteps_other:
                            non_compliance_data = other_data[ts].get("non_compliance", {})
                            
                            # Debug Prints
                            print(f"  Year {ts}: non_compliance_data is: {non_compliance_data}")
                            
                            total_non_compliant = sum(non_compliance_data.values())
                            
                            print(f"  Year {ts}: total_non_compliant: {total_non_compliant} | bond_revenue_added: {total_non_compliant * bond_amount}")
                            
                            bond_revenues.append(total_non_compliant * bond_amount)

                        bond_revenues_array = np.array(bond_revenues)
                        
                        # --- 3. Calculate Total Welfare (Welfare WITH revenue) ---
                        if len(welfare_base) != len(bond_revenues_array):
                            print(f"Warning in '{plot_data.scenario}': Welfare length ({len(welfare_base)}) mismatch with revenue length ({len(bond_revenues_array)}). Skipping revenue.")
                            welfare_total = welfare_base
                        else:
                            welfare_total = welfare_base + bond_revenues_array
                        
                        # --- 4. Plot both lines ---
                        label = getattr(plot_data, 'scenario', 'Unnamed Scenario')
                        time_axis = range(len(welfare_base)) # N+1 steps

                        plt.plot(time_axis, welfare_base, label=f"{label} (Satellites Only)", linewidth=2, linestyle='--')
                        plt.plot(time_axis, welfare_total, label=f"{label} (Satellites + Bond Revenue)", linewidth=2, linestyle='-')

                plt.xlabel("Year", fontsize=12)
                plt.ylabel("Welfare", fontsize=12)
                plt.legend(title="Scenario", fontsize=10, loc='upper left')
                plt.grid(True)
                plt.tight_layout()

                # Save
                outdir = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(outdir, exist_ok=True)
                file_path = os.path.join(outdir, "total_welfare_across_scenarios.png")
                plt.savefig(file_path, dpi=300)
                plt.close()

                print(f"Scenario-wise total welfare plot saved to {file_path}")

        def comparison_object_counts_vs_bond(self, plot_data_lists, other_data_lists):
                """
                Create a comparison line plot across bond levels for:
                - Derelicts split into naturally compliant vs non-compliant using econ_params.naturally_compliant_vector
                - Fringe satellites Su and Sns
                Styling: 5yr PMD = solid lines; 25yr PMD = dashed lines.
                X-axis uses bond amount ($k).
                """
                import os
                import re
                import numpy as np
                import matplotlib.pyplot as plt

                comparison_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(comparison_folder, exist_ok=True)

                # Buckets by lifetime
                data_5yr = {"bond": [], "nat": [], "non": [], "Su": [], "Sns": []}
                data_25yr = {"bond": [], "nat": [], "non": [], "Su": [], "Sns": []}

                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        scenario_label = getattr(plot_data, 'scenario', f"Scenario {i+1}")

                        # Extract bond amount and lifetime
                        m = re.search(r"bond_(\d+)k_(\d+)yr", scenario_label.lower())
                        if not m:
                                continue
                        bond_k = int(m.group(1))
                        lifetime = int(m.group(2))

                        # Final timestep index
                        timesteps = sorted(other_data.keys(), key=int)
                        final_ts = timesteps[-1]

                        # econ_params for naturally_compliant_vector
                        econ = plot_data.econ_params if hasattr(plot_data, 'econ_params') else None
                        nat_vec = np.array(econ.get("naturally_compliant_vector", [])) if econ else np.array([])

                        # Aggregate derelicts across all N_* species at final year per shell
                        derelicts_per_shell = None
                        for sp_name, sp_data in plot_data.data.items():
                                if sp_name.startswith('N'):
                                        arr = np.array(sp_data)
                                        if arr.ndim == 2 and arr.shape[0] > 0:  # (time, shells)
                                                final_row = arr[-1, :]
                                                derelicts_per_shell = final_row if derelicts_per_shell is None else derelicts_per_shell + final_row

                        # If nat_vec length mismatches shells, fall back to totals without split
                        nat_sum = 0.0
                        non_sum = 0.0
                        if derelicts_per_shell is not None and nat_vec.size == derelicts_per_shell.size:
                                nat_sum = float(np.sum(derelicts_per_shell[nat_vec == 1]))
                                non_sum = float(np.sum(derelicts_per_shell[nat_vec == 0]))
                        elif derelicts_per_shell is not None:
                                total_d = float(np.sum(derelicts_per_shell))
                                nat_sum = total_d / 2.0
                                non_sum = total_d / 2.0

                        # Fringe satellites totals at final year
                        def sum_final(species_key: str) -> float:
                                if species_key in plot_data.data:
                                        arr = np.array(plot_data.data[species_key])
                                        if arr.ndim == 2 and arr.shape[0] > 0:
                                                return float(np.sum(arr[-1, :]))
                                return 0.0

                        su_total = sum_final('Su')
                        sns_total = sum_final('Sns')

                        bucket = data_5yr if lifetime == 5 else data_25yr
                        bucket["bond"].append(bond_k)
                        bucket["nat"].append(nat_sum)
                        bucket["non"].append(non_sum)
                        bucket["Su"].append(su_total)
                        bucket["Sns"].append(sns_total)

                # Sort by bond for smooth lines
                def sort_bucket(b):
                        if not b["bond"]:
                                return b
                        order = np.argsort(b["bond"])  # ascending
                        for k in b.keys():
                                b[k] = [b[k][idx] for idx in order]
                        return b

                data_5yr = sort_bucket(data_5yr)
                data_25yr = sort_bucket(data_25yr)

                # Plot
                plt.figure(figsize=(12, 7))

                # 5yr solid
                plt.plot(data_5yr["bond"], data_5yr["nat"], color='green', marker='o', linestyle='-', label='5yr PMD - Nat. Compliant Derelicts')
                plt.plot(data_5yr["bond"], data_5yr["non"], color='green', marker='o', linestyle='-', alpha=0.75, label='5yr PMD - Non-Comp. Derelicts')
                plt.plot(data_5yr["bond"], data_5yr["Su"], color='blue', marker='s', linestyle='-', label='5yr PMD - Fringe Sats (Su)')
                plt.plot(data_5yr["bond"], data_5yr["Sns"], color='purple', marker='x', linestyle='-', label='5yr PMD - Fringe Sats (Sns)')

                # 25yr dashed
                plt.plot(data_25yr["bond"], data_25yr["nat"], color='red', marker='o', linestyle='--', label='25yr PMD - Nat. Compliant Derelicts')
                plt.plot(data_25yr["bond"], data_25yr["non"], color='red', marker='o', linestyle='--', alpha=0.75, label='25yr PMD - Non-Comp. Derelicts')
                plt.plot(data_25yr["bond"], data_25yr["Su"], color='blue', marker='s', linestyle='--', label='25yr PMD - Fringe Sats (Su)')
                plt.plot(data_25yr["bond"], data_25yr["Sns"], color='purple', marker='x', linestyle='--', label='25yr PMD - Fringe Sats (Sns)')

                plt.xlabel("Lifetime Bond Amount, $ (k)", fontsize=14, fontweight='bold')
                plt.ylabel("Number of Objects", fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=11, loc='best')
                plt.tight_layout()

                file_path = os.path.join(comparison_folder, "object_counts_vs_bond_split.png")
                plt.savefig(file_path, dpi=300)
                plt.close()
                print(f"Split object count plot saved to {file_path}")

        def comparison_object_counts_vs_bond_by_species(self, plot_data_lists, other_data_lists):
                """
                Create 2x2 subplots showing:
                - Three plots (S, Su, Sns) with active satellites, compliant derelicts, and non-compliant derelicts
                - One plot showing all debris (excluding derelicts and active satellites)
                All plotted against bond amount.
                """
                import os
                import re
                import numpy as np
                import matplotlib.pyplot as plt

                comparison_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(comparison_folder, exist_ok=True)

                # Store data for each species: S, Su, Sns
                # Each will have bond amounts, active satellites, compliant derelicts, non-compliant derelicts
                species_data = {
                        "S": {"bond": [], "active": [], "compliant_derelicts": [], "non_compliant_derelicts": []},
                        "Su": {"bond": [], "active": [], "compliant_derelicts": [], "non_compliant_derelicts": []},
                        "Sns": {"bond": [], "active": [], "compliant_derelicts": [], "non_compliant_derelicts": []}
                }
                
                # Store data for all debris (excluding derelicts)
                debris_data = {"bond": [], "total_debris_10cm": []}

                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        scenario_label = getattr(plot_data, 'scenario', f"Scenario {i+1}")

                        # Extract bond amount
                        m = re.search(r"bond_(\d+)k", scenario_label.lower())
                        if not m:
                                continue
                        bond_k = int(m.group(1))

                        # Final timestep index
                        timesteps = sorted(other_data.keys(), key=int)
                        final_ts = timesteps[-1]

                        # Get compliance and non-compliance values directly from other_results for final timestep
                        compliance = {}
                        non_compliance = {}
                        if final_ts in other_data:
                                comp_data = other_data[final_ts].get("compliance", {})
                                nc_data = other_data[final_ts].get("non_compliance", {})
                                for sp_key in ["S", "Su", "Sns"]:
                                        compliance[sp_key] = float(comp_data.get(sp_key, 0.0))
                                        non_compliance[sp_key] = float(nc_data.get(sp_key, 0.0))

                        # Get active satellite counts for each species
                        def sum_final(species_key: str) -> float:
                                if species_key in plot_data.data:
                                        arr = np.array(plot_data.data[species_key])
                                        if arr.ndim == 2 and arr.shape[0] > 0:
                                                return float(np.sum(arr[-1, :]))
                                return 0.0

                        s_active = sum_final('S')
                        su_active = sum_final('Su')
                        sns_active = sum_final('Sns')

                        # Use compliance and non-compliance values directly from other_results
                        species_derelicts = {}
                        for species_key in ["S", "Su", "Sns"]:
                                compliant_derelicts = compliance.get(species_key, 0.0)
                                non_compliant_derelicts = non_compliance.get(species_key, 0.0)
                                
                                species_derelicts[species_key] = {
                                        'compliant': compliant_derelicts,
                                        'non_compliant': non_compliant_derelicts
                                }

                        # Store data for each species
                        for species_key in ["S", "Su", "Sns"]:
                                species_data[species_key]["bond"].append(bond_k)
                                species_data[species_key]["compliant_derelicts"].append(species_derelicts[species_key]['compliant'])
                                species_data[species_key]["non_compliant_derelicts"].append(species_derelicts[species_key]['non_compliant'])
                                
                                if species_key == "S":
                                        species_data[species_key]["active"].append(s_active)
                                elif species_key == "Su":
                                        species_data[species_key]["active"].append(su_active)
                                else:  # Sns
                                        species_data[species_key]["active"].append(sns_active)
                        
                        # Calculate total debris (excluding derelicts and active satellites)
                        total_debris = 0.0
                        
                        # Get active species names and derelict species names
                        active_species = ["S", "Su", "Sns"]
                        derelict_species_names = getattr(plot_data, 'derelict_species_names', [])
                        if not derelict_species_names:
                                # Fallback: derelicts start with 'N' or 'B'
                                derelict_species_names = [name for name in plot_data.species_names 
                                                          if name.startswith('N') or name.startswith('B')]
                        
                        # Sum all debris species (not active, not derelicts)
                        for sp_name in plot_data.species_names:
                                # Skip active species and derelicts
                                if sp_name in active_species or sp_name in derelict_species_names:
                                        continue
                                
                                # Get count for this debris species at final timestep
                                if sp_name in plot_data.data:
                                        arr = np.array(plot_data.data[sp_name])
                                        if arr.ndim == 2 and arr.shape[0] > 0:
                                                count = float(np.sum(arr[-1, :]))
                                                total_debris += count
                        
                        debris_data["bond"].append(bond_k)
                        debris_data["total_debris_10cm"].append(total_debris)

                # Sort by bond for smooth lines
                def sort_species_data(data_dict):
                        if not data_dict["bond"]:
                                return data_dict
                        order = np.argsort(data_dict["bond"])  # ascending
                        for k in data_dict.keys():
                                data_dict[k] = [data_dict[k][idx] for idx in order]
                        return data_dict

                for species_key in species_data.keys():
                        species_data[species_key] = sort_species_data(species_data[species_key])
                
                # Sort debris data
                if debris_data["bond"]:
                        order = np.argsort(debris_data["bond"])
                        debris_data["bond"] = [debris_data["bond"][idx] for idx in order]
                        debris_data["total_debris_10cm"] = [debris_data["total_debris_10cm"][idx] for idx in order]

                # Create 2x2 subplots
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                axes = axes.flatten()  # Flatten to 1D array for easier indexing
                species_names = ["S", "Su", "Sns"]
                
                # Consistent colors across all graphs
                active_color = "blue"
                compliant_color = "green"
                non_compliant_color = "red"
                debris_color = "orange"

                # Plot first three subplots for S, Su, Sns
                for idx, species_key in enumerate(species_names):
                        ax = axes[idx]
                        data = species_data[species_key]

                        # Plot active satellites (blue)
                        ax.plot(data["bond"], data["active"], color=active_color, marker='s', linestyle='-', 
                               linewidth=2, markersize=6, label=f'Active {species_key} Satellites', zorder=3)
                        
                        # Plot compliant derelicts (green)
                        ax.plot(data["bond"], data["compliant_derelicts"], color=compliant_color, marker='o', 
                               linestyle='-', linewidth=2, markersize=6, label='Compliant Derelicts', zorder=2)
                        
                        # Plot non-compliant derelicts (red)
                        ax.plot(data["bond"], data["non_compliant_derelicts"], color=non_compliant_color, marker='x', 
                               linestyle='-', linewidth=2, markersize=6, label='Non-Compliant Derelicts', zorder=2)

                        ax.set_xlabel("Lifetime Bond Amount, $ (k)", fontsize=14, fontweight='bold')
                        ax.set_ylabel("Number of Objects", fontsize=14, fontweight='bold')
                        ax.set_title(f"{species_key}", fontsize=16, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        ax.legend(fontsize=11, loc='best')
                        
                        # Make ticks bold
                        ax.tick_params(axis='both', which='major', labelsize=11, width=1.5)
                        for label in ax.get_xticklabels():
                                label.set_fontweight('bold')
                        for label in ax.get_yticklabels():
                                label.set_fontweight('bold')
                
                # Plot fourth subplot for all debris (excluding derelicts)
                ax = axes[3]
                ax.plot(debris_data["bond"], debris_data["total_debris_10cm"], color=debris_color, marker='^', 
                       linestyle='-', linewidth=2, markersize=6, label='All Debris', zorder=3)
                
                ax.set_xlabel("Lifetime Bond Amount, $ (k)", fontsize=14, fontweight='bold')
                ax.set_ylabel("Number of Objects", fontsize=14, fontweight='bold')
                ax.set_title("All Debris", fontsize=16, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=11, loc='best')
                
                # Make ticks bold
                ax.tick_params(axis='both', which='major', labelsize=11, width=1.5)
                for label in ax.get_xticklabels():
                        label.set_fontweight('bold')
                for label in ax.get_yticklabels():
                        label.set_fontweight('bold')

                plt.tight_layout()

                file_path = os.path.join(comparison_folder, "object_counts_vs_bond_by_species.png")
                plt.savefig(file_path, dpi=300)
                plt.close()
                print(f"Object counts by species plot saved to {file_path}")

        def comparison_object_counts_vs_bond_by_species_with_umpy(self, plot_data_lists, other_data_lists):
                """
                Create 2x2 subplots showing:
                - Three plots (S, Su, Sns) with active satellites, compliant derelicts, and non-compliant derelicts
                - One plot showing final year UMPY vs bond amount
                All plotted against bond amount.
                """
                import os
                import re
                import numpy as np
                import matplotlib.pyplot as plt

                comparison_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(comparison_folder, exist_ok=True)

                # Store data for each species: S, Su, Sns
                # Each will have bond amounts, active satellites, compliant derelicts, non-compliant derelicts
                species_data = {
                        "S": {"bond": [], "active": [], "compliant_derelicts": [], "non_compliant_derelicts": []},
                        "Su": {"bond": [], "active": [], "compliant_derelicts": [], "non_compliant_derelicts": []},
                        "Sns": {"bond": [], "active": [], "compliant_derelicts": [], "non_compliant_derelicts": []}
                }
                
                # Store data for final year UMPY
                umpy_data = {"bond": [], "final_umpy": []}

                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        scenario_label = getattr(plot_data, 'scenario', f"Scenario {i+1}")

                        # Extract bond amount
                        m = re.search(r"bond_(\d+)k", scenario_label.lower())
                        if not m:
                                continue
                        bond_k = int(m.group(1))

                        # Final timestep index
                        timesteps = sorted(other_data.keys(), key=int)
                        final_ts = timesteps[-1]

                        # Get compliance and non-compliance values directly from other_results for final timestep
                        compliance = {}
                        non_compliance = {}
                        if final_ts in other_data:
                                comp_data = other_data[final_ts].get("compliance", {})
                                nc_data = other_data[final_ts].get("non_compliance", {})
                                for sp_key in ["S", "Su", "Sns"]:
                                        compliance[sp_key] = float(comp_data.get(sp_key, 0.0))
                                        non_compliance[sp_key] = float(nc_data.get(sp_key, 0.0))

                        # Get active satellite counts for each species
                        def sum_final(species_key: str) -> float:
                                if species_key in plot_data.data:
                                        arr = np.array(plot_data.data[species_key])
                                        if arr.ndim == 2 and arr.shape[0] > 0:
                                                return float(np.sum(arr[-1, :]))
                                return 0.0

                        s_active = sum_final('S')
                        su_active = sum_final('Su')
                        sns_active = sum_final('Sns')

                        # Use compliance and non-compliance values directly from other_results
                        species_derelicts = {}
                        for species_key in ["S", "Su", "Sns"]:
                                compliant_derelicts = compliance.get(species_key, 0.0)
                                non_compliant_derelicts = non_compliance.get(species_key, 0.0)
                                
                                species_derelicts[species_key] = {
                                        'compliant': compliant_derelicts,
                                        'non_compliant': non_compliant_derelicts
                                }

                        # Store data for each species
                        for species_key in ["S", "Su", "Sns"]:
                                species_data[species_key]["bond"].append(bond_k)
                                species_data[species_key]["compliant_derelicts"].append(species_derelicts[species_key]['compliant'])
                                species_data[species_key]["non_compliant_derelicts"].append(species_derelicts[species_key]['non_compliant'])
                                
                                if species_key == "S":
                                        species_data[species_key]["active"].append(s_active)
                                elif species_key == "Su":
                                        species_data[species_key]["active"].append(su_active)
                                else:  # Sns
                                        species_data[species_key]["active"].append(sns_active)
                        
                        # Get final year UMPY
                        final_umpy = 0.0
                        if final_ts in other_data:
                                umpy_data_ts = other_data[final_ts].get("umpy", [])
                                if isinstance(umpy_data_ts, (list, np.ndarray)):
                                        # If it's a list/array, sum all values
                                        final_umpy = float(np.sum(umpy_data_ts))
                                elif isinstance(umpy_data_ts, dict):
                                        # If it's a dict, sum all values
                                        final_umpy = float(np.sum(list(umpy_data_ts.values())))
                                elif isinstance(umpy_data_ts, (int, float)):
                                        # If it's a single value
                                        final_umpy = float(umpy_data_ts)
                        
                        umpy_data["bond"].append(bond_k)
                        umpy_data["final_umpy"].append(final_umpy)

                # Sort by bond for smooth lines
                def sort_species_data(data_dict):
                        if not data_dict["bond"]:
                                return data_dict
                        order = np.argsort(data_dict["bond"])  # ascending
                        for k in data_dict.keys():
                                data_dict[k] = [data_dict[k][idx] for idx in order]
                        return data_dict

                for species_key in species_data.keys():
                        species_data[species_key] = sort_species_data(species_data[species_key])
                
                # Sort UMPY data
                if umpy_data["bond"]:
                        order = np.argsort(umpy_data["bond"])
                        umpy_data["bond"] = [umpy_data["bond"][idx] for idx in order]
                        umpy_data["final_umpy"] = [umpy_data["final_umpy"][idx] for idx in order]

                # Create 2x2 subplots
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                axes = axes.flatten()  # Flatten to 1D array for easier indexing
                species_names = ["S", "Su", "Sns"]
                
                # Consistent colors across all graphs
                active_color = "blue"
                compliant_color = "green"
                non_compliant_color = "red"
                umpy_color = "purple"

                # Plot first three subplots for S, Su, Sns
                for idx, species_key in enumerate(species_names):
                        ax = axes[idx]
                        data = species_data[species_key]

                        # Plot active satellites (blue)
                        ax.plot(data["bond"], data["active"], color=active_color, marker='s', linestyle='-', 
                               linewidth=2, markersize=6, label=f'Active {species_key} Satellites', zorder=3)
                        
                        # Plot compliant derelicts (green)
                        ax.plot(data["bond"], data["compliant_derelicts"], color=compliant_color, marker='o', 
                               linestyle='-', linewidth=2, markersize=6, label='Compliant Derelicts', zorder=2)
                        
                        # Plot non-compliant derelicts (red)
                        ax.plot(data["bond"], data["non_compliant_derelicts"], color=non_compliant_color, marker='x', 
                               linestyle='-', linewidth=2, markersize=6, label='Non-Compliant Derelicts', zorder=2)

                        ax.set_xlabel("Lifetime Bond Amount, $ (k)", fontsize=14, fontweight='bold')
                        ax.set_ylabel("Number of Objects", fontsize=14, fontweight='bold')
                        ax.set_title(f"{species_key}", fontsize=16, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        ax.legend(fontsize=11, loc='best')
                        
                        # Make ticks bold
                        ax.tick_params(axis='both', which='major', labelsize=11, width=1.5)
                        for label in ax.get_xticklabels():
                                label.set_fontweight('bold')
                        for label in ax.get_yticklabels():
                                label.set_fontweight('bold')
                
                # Plot fourth subplot for final year UMPY
                ax = axes[3]
                ax.plot(umpy_data["bond"], umpy_data["final_umpy"], color=umpy_color, marker='^', 
                       linestyle='-', linewidth=2, markersize=6, label='Final Year UMPY', zorder=3)
                
                ax.set_xlabel("Lifetime Bond Amount, $ (k)", fontsize=14, fontweight='bold')
                ax.set_ylabel("UMPY (kg/year)", fontsize=14, fontweight='bold')
                ax.set_title("Final Year UMPY", fontsize=16, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=11, loc='best')
                
                # Make ticks bold
                ax.tick_params(axis='both', which='major', labelsize=11, width=1.5)
                for label in ax.get_xticklabels():
                        label.set_fontweight('bold')
                for label in ax.get_yticklabels():
                        label.set_fontweight('bold')

                plt.tight_layout()

                file_path = os.path.join(comparison_folder, "object_counts_vs_bond_by_species_with_umpy.png")
                plt.savefig(file_path, dpi=300)
                plt.close()
                print(f"Object counts by species with UMPY plot saved to {file_path}")

        def comparison_object_counts_vs_bond_by_species_with_umpy_relative(self, plot_data_lists, other_data_lists):
                """
                Create 2x2 subplots showing:
                - Three plots (S, Su, Sns) with active satellites, compliant derelicts, and non-compliant derelicts
                - One plot showing relative change in final year UMPY from baseline (%) vs bond amount
                All plotted against bond amount.
                """
                import os
                import re
                import numpy as np
                import matplotlib.pyplot as plt

                comparison_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(comparison_folder, exist_ok=True)

                # Store data for each species: S, Su, Sns
                # Each will have bond amounts, active satellites, compliant derelicts, non-compliant derelicts
                species_data = {
                        "S": {"bond": [], "active": [], "compliant_derelicts": [], "non_compliant_derelicts": []},
                        "Su": {"bond": [], "active": [], "compliant_derelicts": [], "non_compliant_derelicts": []},
                        "Sns": {"bond": [], "active": [], "compliant_derelicts": [], "non_compliant_derelicts": []}
                }
                
                # Store data for relative change in final year UMPY
                umpy_relative_data = {"bond": [], "relative_change_percent": []}

                # First pass: find baseline UMPY (bond_0k scenario)
                baseline_umpy = None
                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        scenario_label = getattr(plot_data, 'scenario', f"Scenario {i+1}")
                        
                        # Check if this is the baseline (bond_0k)
                        m = re.search(r"bond_(\d+)k", scenario_label.lower())
                        if m and int(m.group(1)) == 0:
                                # This is the baseline scenario
                                timesteps = sorted(other_data.keys(), key=int)
                                final_ts = timesteps[-1]
                                
                                if final_ts in other_data:
                                        umpy_data_ts = other_data[final_ts].get("umpy", [])
                                        if isinstance(umpy_data_ts, (list, np.ndarray)):
                                                baseline_umpy = float(np.sum(umpy_data_ts))
                                        elif isinstance(umpy_data_ts, dict):
                                                baseline_umpy = float(np.sum(list(umpy_data_ts.values())))
                                        elif isinstance(umpy_data_ts, (int, float)):
                                                baseline_umpy = float(umpy_data_ts)
                                break
                
                # If baseline not found, use first scenario as baseline
                if baseline_umpy is None and plot_data_lists:
                        plot_data, other_data = plot_data_lists[0], other_data_lists[0]
                        timesteps = sorted(other_data.keys(), key=int)
                        final_ts = timesteps[-1]
                        if final_ts in other_data:
                                umpy_data_ts = other_data[final_ts].get("umpy", [])
                                if isinstance(umpy_data_ts, (list, np.ndarray)):
                                        baseline_umpy = float(np.sum(umpy_data_ts))
                                elif isinstance(umpy_data_ts, dict):
                                        baseline_umpy = float(np.sum(list(umpy_data_ts.values())))
                                elif isinstance(umpy_data_ts, (int, float)):
                                        baseline_umpy = float(umpy_data_ts)

                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        scenario_label = getattr(plot_data, 'scenario', f"Scenario {i+1}")

                        # Extract bond amount
                        m = re.search(r"bond_(\d+)k", scenario_label.lower())
                        if not m:
                                continue
                        bond_k = int(m.group(1))

                        # Final timestep index
                        timesteps = sorted(other_data.keys(), key=int)
                        final_ts = timesteps[-1]

                        # Get compliance and non-compliance values directly from other_results for final timestep
                        compliance = {}
                        non_compliance = {}
                        if final_ts in other_data:
                                comp_data = other_data[final_ts].get("compliance", {})
                                nc_data = other_data[final_ts].get("non_compliance", {})
                                for sp_key in ["S", "Su", "Sns"]:
                                        compliance[sp_key] = float(comp_data.get(sp_key, 0.0))
                                        non_compliance[sp_key] = float(nc_data.get(sp_key, 0.0))

                        # Get active satellite counts for each species
                        def sum_final(species_key: str) -> float:
                                if species_key in plot_data.data:
                                        arr = np.array(plot_data.data[species_key])
                                        if arr.ndim == 2 and arr.shape[0] > 0:
                                                return float(np.sum(arr[-1, :]))
                                return 0.0

                        s_active = sum_final('S')
                        su_active = sum_final('Su')
                        sns_active = sum_final('Sns')

                        # Use compliance and non-compliance values directly from other_results
                        species_derelicts = {}
                        for species_key in ["S", "Su", "Sns"]:
                                compliant_derelicts = compliance.get(species_key, 0.0)
                                non_compliant_derelicts = non_compliance.get(species_key, 0.0)
                                
                                species_derelicts[species_key] = {
                                        'compliant': compliant_derelicts,
                                        'non_compliant': non_compliant_derelicts
                                }

                        # Store data for each species
                        for species_key in ["S", "Su", "Sns"]:
                                species_data[species_key]["bond"].append(bond_k)
                                species_data[species_key]["compliant_derelicts"].append(species_derelicts[species_key]['compliant'])
                                species_data[species_key]["non_compliant_derelicts"].append(species_derelicts[species_key]['non_compliant'])
                                
                                if species_key == "S":
                                        species_data[species_key]["active"].append(s_active)
                                elif species_key == "Su":
                                        species_data[species_key]["active"].append(su_active)
                                else:  # Sns
                                        species_data[species_key]["active"].append(sns_active)
                        
                        # Get final year UMPY and calculate relative change
                        final_umpy = 0.0
                        if final_ts in other_data:
                                umpy_data_ts = other_data[final_ts].get("umpy", [])
                                if isinstance(umpy_data_ts, (list, np.ndarray)):
                                        # If it's a list/array, sum all values
                                        final_umpy = float(np.sum(umpy_data_ts))
                                elif isinstance(umpy_data_ts, dict):
                                        # If it's a dict, sum all values
                                        final_umpy = float(np.sum(list(umpy_data_ts.values())))
                                elif isinstance(umpy_data_ts, (int, float)):
                                        # If it's a single value
                                        final_umpy = float(umpy_data_ts)
                        
                        # Calculate relative change as percentage: ((umpy - baseline) / baseline) * 100
                        if baseline_umpy is not None and baseline_umpy > 0:
                                relative_change = ((final_umpy - baseline_umpy) / baseline_umpy) * 100
                        else:
                                relative_change = 0.0
                        
                        umpy_relative_data["bond"].append(bond_k)
                        umpy_relative_data["relative_change_percent"].append(relative_change)

                # Sort by bond for smooth lines
                def sort_species_data(data_dict):
                        if not data_dict["bond"]:
                                return data_dict
                        order = np.argsort(data_dict["bond"])  # ascending
                        for k in data_dict.keys():
                                data_dict[k] = [data_dict[k][idx] for idx in order]
                        return data_dict

                for species_key in species_data.keys():
                        species_data[species_key] = sort_species_data(species_data[species_key])
                
                # Sort UMPY relative data
                if umpy_relative_data["bond"]:
                        order = np.argsort(umpy_relative_data["bond"])
                        umpy_relative_data["bond"] = [umpy_relative_data["bond"][idx] for idx in order]
                        umpy_relative_data["relative_change_percent"] = [umpy_relative_data["relative_change_percent"][idx] for idx in order]

                # Create 2x2 subplots
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                axes = axes.flatten()  # Flatten to 1D array for easier indexing
                species_names = ["S", "Su", "Sns"]
                
                # Consistent colors across all graphs
                active_color = "blue"
                compliant_color = "green"
                non_compliant_color = "red"
                umpy_relative_color = "purple"

                # Plot first three subplots for S, Su, Sns
                for idx, species_key in enumerate(species_names):
                        ax = axes[idx]
                        data = species_data[species_key]
                        
                        # Convert bond amounts from k to M for plotting
                        bond_m = [b / 1000.0 for b in data["bond"]]

                        # Plot active satellites (blue)
                        ax.plot(bond_m, data["active"], color=active_color, marker='s', linestyle='-', 
                               linewidth=2, markersize=6, label=f'Active {species_key} Satellites', zorder=3)
                        
                        # Plot compliant derelicts (green)
                        ax.plot(bond_m, data["compliant_derelicts"], color=compliant_color, marker='o', 
                               linestyle='-', linewidth=2, markersize=6, label='Compliant Derelicts', zorder=2)
                        
                        # Plot non-compliant derelicts (red)
                        ax.plot(bond_m, data["non_compliant_derelicts"], color=non_compliant_color, marker='x', 
                               linestyle='-', linewidth=2, markersize=6, label='Non-Compliant Derelicts', zorder=2)

                        ax.set_xlabel("Lifetime Bond Amount, $ (M)", fontsize=14, fontweight='bold')
                        ax.set_ylabel("Number of Objects", fontsize=14, fontweight='bold')
                        ax.set_title(f"{species_key}", fontsize=16, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        ax.legend(fontsize=11, loc='best')
                        
                        # Make ticks bold
                        ax.tick_params(axis='both', which='major', labelsize=11, width=1.5)
                        for label in ax.get_xticklabels():
                                label.set_fontweight('bold')
                        for label in ax.get_yticklabels():
                                label.set_fontweight('bold')
                
                # Plot fourth subplot for relative change in final year UMPY
                ax = axes[3]
                # Convert bond amounts from k to M for plotting
                bond_m = [b / 1000.0 for b in umpy_relative_data["bond"]]
                ax.plot(bond_m, umpy_relative_data["relative_change_percent"], color=umpy_relative_color, marker='^', 
                       linestyle='-', linewidth=2, markersize=6, label='Relative Change in UMPY', zorder=3)
                
                # Add horizontal line at 0% for baseline
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
                
                ax.set_xlabel("Lifetime Bond Amount, $ (M)", fontsize=14, fontweight='bold')
                ax.set_ylabel("Relative Change in UMPY (%)", fontsize=14, fontweight='bold')
                ax.set_title("Relative UMPY difference to Baseline", fontsize=16, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=13, loc='upper right')
                
                # Make ticks bold and increase size by 2
                ax.tick_params(axis='both', which='major', labelsize=13, width=1.5)
                for label in ax.get_xticklabels():
                        label.set_fontweight('bold')
                for label in ax.get_yticklabels():
                        label.set_fontweight('bold')

                plt.tight_layout()

                file_path = os.path.join(comparison_folder, "object_counts_vs_bond_by_species_with_umpy_relative.png")
                plt.savefig(file_path, dpi=300)
                plt.close()
                print(f"Object counts by species with UMPY relative change plot saved to {file_path}")

        def comparison_scatter_umpy_vs_total_objects(self, plot_data_lists, other_data_lists):
                """
                Create a scatter plot showing FINAL YEAR ONLY:
                - X-axis: Total UMPY across all shells
                - Y-axis: Total count of objects
                - Color: Bond amount
                """
                import matplotlib.pyplot as plt
                import numpy as np
                import os
                import re

                # Create comparison folder
                comparison_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(comparison_folder, exist_ok=True)
                file_path = os.path.join(comparison_folder, "scatter_umpy_vs_total_objects.png")

                umpy_vals = []
                total_object_vals = []
                bond_vals = []
                scenario_labels = []

                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        scenario_label = getattr(plot_data, 'scenario', f"Scenario {i+1}")
                        
                        # Extract bond amount from scenario name
                        match = re.search(r"bond_(\d+)k", scenario_label.lower())
                        bond = int(match.group(1)) if match else 0
                        
                        timesteps = sorted(other_data.keys(), key=int)
                        final_timestep = timesteps[-1]  # Only final year
                        
                        # Calculate total UMPY for final timestep
                        umpy_data = other_data[final_timestep].get("umpy", [])
                        if isinstance(umpy_data, dict):
                                total_umpy = np.sum(list(umpy_data.values()))
                        else:
                                total_umpy = np.sum(umpy_data) if umpy_data else 0
                        
                        # Calculate total objects across all species for final timestep
                        total_objects = 0
                        for species_name, species_data in plot_data.data.items():
                                if isinstance(species_data, np.ndarray):
                                        # Data is already converted to numpy array (time, shells)
                                        if species_data.ndim == 2:
                                                # Get data for final timestep
                                                total_objects += np.sum(species_data[-1, :])
                                        else:
                                                total_objects += np.sum(species_data)
                                elif isinstance(species_data, dict):
                                        # Data is stored as {year: [shell1, shell2, ...], ...}
                                        year_keys = sorted(species_data.keys(), key=int)
                                        final_year = year_keys[-1]
                                        final_year_data = species_data[final_year]
                                        if isinstance(final_year_data, list):
                                                total_objects += np.sum(final_year_data)
                                elif isinstance(species_data, list) and len(species_data) > 0:
                                        arr = np.array(species_data)
                                        if arr.ndim == 2:  # (time, shells)
                                                # Get data for final timestep
                                                total_objects += np.sum(arr[-1, :])
                                        else:
                                                total_objects += np.sum(arr)
                        
                        print(f"Scenario {scenario_label}: UMPY={total_umpy:.2f}, Total Objects={total_objects:.2f}")
                        
                        umpy_vals.append(total_umpy)
                        total_object_vals.append(total_objects)
                        bond_vals.append(bond)
                        scenario_labels.append(scenario_label)

                # Check if we have data
                if not umpy_vals:
                        print("Warning: No data found for scatter_umpy_vs_total_objects plot")
                        return

                # Create the scatter plot
                plt.figure(figsize=(10, 8))
                
                # Create scatter plot with color based on bond amount
                scatter = plt.scatter(umpy_vals, total_object_vals, 
                                    c=bond_vals, s=100, 
                                    cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)
                
                # Add colorbar for bond amounts
                cbar = plt.colorbar(scatter)
                cbar.set_label("Bond Amount ($k)", fontsize=12)
                
                # Add annotations for scenario labels
                for x, y, scenario in zip(umpy_vals, total_object_vals, scenario_labels):
                        clean_label = scenario.replace('bond_', '').replace('_5yr', '').replace('_25yr', '')
                        plt.annotate(clean_label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

                plt.xlabel("Total UMPY (kg/year)", fontsize=14, fontweight='bold')
                plt.ylabel("Total Count of Objects", fontsize=14, fontweight='bold')
                plt.title("Final Year: Total Objects vs UMPY by Bond Amount", fontsize=16, fontweight='bold')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"UMPY vs total objects scatter plot saved to {file_path}")



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
                # plt.title("UMPY Evolution Over Time")  # Removed overall title
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
                        # Ensure the number of labels matches the number of ticks
                        if len(self.HMid) == data.shape[1]:
                            plt.yticks(ticks=range(data.shape[1]), labels=self.HMid)
                        else:
                            # If HMid length doesn't match, use a subset or create appropriate labels
                            if len(self.HMid) < data.shape[1]:
                                # Pad with additional values or use the available ones
                                labels = list(self.HMid) + [f"Shell {i}" for i in range(len(self.HMid), data.shape[1])]
                                plt.yticks(ticks=range(data.shape[1]), labels=labels[:data.shape[1]])
                            else:
                                # Use only the first data.shape[1] elements
                                plt.yticks(ticks=range(data.shape[1]), labels=self.HMid[:data.shape[1]])

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
                        # Ensure the number of labels matches the number of ticks
                        if len(self.HMid) == data.shape[1]:
                            ax.set_yticklabels(self.HMid)
                        else:
                            # If HMid length doesn't match, use a subset or create appropriate labels
                            if len(self.HMid) < data.shape[1]:
                                # Pad with additional values or use the available ones
                                labels = list(self.HMid) + [f"Shell {i}" for i in range(len(self.HMid), data.shape[1])]
                                ax.set_yticklabels(labels[:data.shape[1]])
                            else:
                                # Use only the first data.shape[1] elements
                                ax.set_yticklabels(self.HMid[:data.shape[1]])
                        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

                # Turn off unused subplots
                for ax in axes[len(species_data):]:
                        ax.axis('off')

                # Save the combined plot
                combined_file_path = os.path.join(plot_data.path, "combined_species_heatmaps.png")
                plt.savefig(combined_file_path, dpi=300, bbox_inches='tight')
                plt.close()
        
        def total_objects_over_time(self, plot_data, other_data):
                """
                Creates a plot showing the sum of each species type over time.
                Groups species by their prefixes (S, Sns, Su, N, B) and includes derelicts as 'D'.
                """
                plt.figure(figsize=(12, 8))
                
                # Group species by their prefixes
                species_groups = {
                        'S': [],
                        'Sns': [], 
                        'Su': [],
                        'N': [],
                        'B': []
                }
                
                # Categorize species by their prefixes
                for species_name in plot_data.species_names:
                        if species_name.startswith('S') and not species_name.startswith('Sns') and not species_name.startswith('Su'):
                                species_groups['S'].append(species_name)
                        elif species_name.startswith('Sns'):
                                species_groups['Sns'].append(species_name)
                        elif species_name.startswith('Su'):
                                species_groups['Su'].append(species_name)
                        elif species_name.startswith('N'):
                                species_groups['N'].append(species_name)
                        elif species_name.startswith('B'):
                                species_groups['B'].append(species_name)
                
                # Calculate totals for each group
                group_totals = {}
                
                for group_name, species_list in species_groups.items():
                        if species_list:  # Only process if there are species in this group
                                # Get the first species to determine time dimension
                                first_species_data = np.array(plot_data.data[species_list[0]])
                                total_for_group = np.zeros(first_species_data.shape[0])  # Initialize with time dimension
                                
                                for species_name in species_list:
                                        if species_name in plot_data.data:
                                                species_data = np.array(plot_data.data[species_name])  # (time, shells)
                                                total_for_group += np.sum(species_data, axis=1)  # Sum across shells
                                
                                group_totals[group_name] = total_for_group
                
                # Add derelicts as 'D' if they exist
                if hasattr(plot_data, 'derelict_species_names') and plot_data.derelict_species_names:
                        # Get time dimension from first available species
                        first_species_key = list(plot_data.data.keys())[0]
                        first_species_data = np.array(plot_data.data[first_species_key])
                        derelict_total = np.zeros(first_species_data.shape[0])
                        
                        for derelict_name in plot_data.derelict_species_names:
                                if derelict_name in plot_data.data:
                                        derelict_data = np.array(plot_data.data[derelict_name])
                                        derelict_total += np.sum(derelict_data, axis=1)
                        
                        group_totals['D'] = derelict_total
                
                # Create time array (assuming equal time steps)
                time_steps = np.arange(len(list(group_totals.values())[0]))
                
                # Plot each group
                colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
                for i, (group_name, total_data) in enumerate(group_totals.items()):
                        if len(total_data) > 0:  # Only plot if there's data
                                plt.plot(time_steps, total_data, label=f'{group_name} (Total)', 
                                        color=colors[i % len(colors)], linewidth=2)
                
                # Calculate and plot grand total
                grand_total = np.zeros_like(list(group_totals.values())[0])
                # Updated logic
                for group_name, total_data in group_totals.items():
                        if group_name != 'D': # Exclude the 'D' group from the grand total
                                grand_total += total_data
                
                plt.plot(time_steps, grand_total, label='Grand Total', 
                        color='black', linewidth=3, linestyle='--')
                
                plt.xlabel('Time Steps')
                plt.ylabel('Total Number of Objects')
                plt.title('Total Objects Over Time by Species Group')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save the plot
                file_path = os.path.join(plot_data.path, 'total_objects_over_time.png')
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Total objects over time plot saved to {file_path}")

        def comparison_total_objects_over_time(self, plot_data_lists, other_data_lists):
                """
                Creates a comparison plot showing the sum of each species type over time across scenarios.
                Groups species by their prefixes and shows them in separate subplots:
                - Active objects (S, Su, Sns)
                - Debris (N, B) 
                - Derelicts (D)
                """
                # Create a "comparisons" folder under the main simulation folder
                comparison_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(comparison_folder, exist_ok=True)
                
                # Create figure with 3 subplots
                fig, axes = plt.subplots(3, 1, figsize=(12, 15))
                
                # Define groups
                active_groups = ['S', 'Su', 'Sns']
                debris_groups = ['N', 'B']
                derelict_groups = ['D']
                
                # Colors for different scenarios
                scenario_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
                
                # Process each scenario
                for i, plot_data in enumerate(plot_data_lists):
                        scenario_name = getattr(plot_data, "scenario", f"Scenario {i+1}")
                        color = scenario_colors[i % len(scenario_colors)]
                        
                        # Group species by their prefixes
                        species_groups = {
                                'S': [],
                                'Sns': [], 
                                'Su': [],
                                'N': [],
                                'B': []
                        }
                        
                        # Categorize species by their prefixes
                        for species_name in plot_data.species_names:
                                if species_name.startswith('S') and not species_name.startswith('Sns') and not species_name.startswith('Su'):
                                        species_groups['S'].append(species_name)
                                elif species_name.startswith('Sns'):
                                        species_groups['Sns'].append(species_name)
                                elif species_name.startswith('Su'):
                                        species_groups['Su'].append(species_name)
                                elif species_name.startswith('N'):
                                        species_groups['N'].append(species_name)
                                elif species_name.startswith('B'):
                                        species_groups['B'].append(species_name)
                        
                        # Calculate totals for each group
                        group_totals = {}
                        
                        for group_name, species_list in species_groups.items():
                                if species_list:  # Only process if there are species in this group
                                        # Get the first species to determine time dimension
                                        first_species_data = np.array(plot_data.data[species_list[0]])
                                        total_for_group = np.zeros(first_species_data.shape[0])  # Initialize with time dimension
                                        
                                        for species_name in species_list:
                                                if species_name in plot_data.data:
                                                        species_data = np.array(plot_data.data[species_name])  # (time, shells)
                                                        total_for_group += np.sum(species_data, axis=1)  # Sum across shells
                                        
                                        group_totals[group_name] = total_for_group
                        
                        # Add derelicts as 'D' if they exist
                        if hasattr(plot_data, 'derelict_species_names') and plot_data.derelict_species_names:
                                # Get time dimension from first available species
                                first_species_key = list(plot_data.data.keys())[0]
                                first_species_data = np.array(plot_data.data[first_species_key])
                                derelict_total = np.zeros(first_species_data.shape[0])
                                
                                for derelict_name in plot_data.derelict_species_names:
                                        if derelict_name in plot_data.data:
                                                derelict_data = np.array(plot_data.data[derelict_name])
                                                derelict_total += np.sum(derelict_data, axis=1)
                                
                                group_totals['D'] = derelict_total
                        
                        # Create time array
                        if group_totals:
                                time_steps = np.arange(len(list(group_totals.values())[0]))
                                
                                # Plot 1: Active objects (S, Su, Sns)
                                active_total = np.zeros_like(time_steps, dtype=float)
                                for group in active_groups:
                                        if group in group_totals:
                                                active_total += group_totals[group]
                                if np.any(active_total > 0):
                                        axes[0].plot(time_steps, active_total, label=f'{scenario_name} - Active', 
                                                  color=color, linewidth=2)
                                
                                # Plot 2: Debris (N, B)
                                debris_total = np.zeros_like(time_steps, dtype=float)
                                for group in debris_groups:
                                        if group in group_totals:
                                                debris_total += group_totals[group]
                                if np.any(debris_total > 0):
                                        axes[1].plot(time_steps, debris_total, label=f'{scenario_name} - Debris', 
                                                  color=color, linewidth=2)
                                
                                # Plot 3: Derelicts (D)
                                if 'D' in group_totals and np.any(group_totals['D'] > 0):
                                        axes[2].plot(time_steps, group_totals['D'], label=f'{scenario_name} - Derelicts', 
                                                  color=color, linewidth=2)
                
                # Configure subplots
                axes[0].set_title('Active Objects (S, Su, Sns) Over Time', fontsize=14, fontweight='bold')
                axes[0].set_ylabel('Total Number of Objects')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                axes[1].set_title('Debris Objects (N, B) Over Time', fontsize=14, fontweight='bold')
                axes[1].set_ylabel('Total Number of Objects')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                axes[2].set_title('Derelict Objects (D) Over Time', fontsize=14, fontweight='bold')
                axes[2].set_xlabel('Time Steps')
                axes[2].set_ylabel('Total Number of Objects')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save the plot
                file_path = os.path.join(comparison_folder, 'comparison_total_objects_over_time.png')
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Comparison total objects over time plot saved to {file_path}")


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

        def _create_3d_maneuver_plots(self, plot_data, other_data):
                """
                Create 3D maneuver plots for each species over time.
                """
                # Create maneuvers folder
                maneuvers_dir = os.path.join(plot_data.path, "maneuvers")
                os.makedirs(maneuvers_dir, exist_ok=True)
                
                try:
                        # Get time steps
                        timesteps = sorted(other_data.keys(), key=int)
                        
                        # Create 3D plot for each species
                        for species_name in plot_data.species_names:
                                if species_name.startswith('S'):  # Only active species
                                        # Extract maneuver data over time
                                        maneuver_data = []
                                        for timestep in timesteps:
                                                if 'maneuvers' in other_data[timestep] and other_data[timestep]['maneuvers'] is not None:
                                                        if isinstance(other_data[timestep]['maneuvers'], dict) and species_name in other_data[timestep]['maneuvers']:
                                                                maneuver_data.append(other_data[timestep]['maneuvers'][species_name])
                                                        else:
                                                                maneuver_data.append([0] * self.n_shells)  # Default to zeros
                                                else:
                                                        maneuver_data.append([0] * self.n_shells)  # Default to zeros
                                        
                                        if maneuver_data:
                                                # Convert to numpy array
                                                maneuver_array = np.array(maneuver_data)
                                                
                                                # Create 3D plot
                                                fig = plt.figure(figsize=(12, 10))
                                                ax = fig.add_subplot(111, projection='3d')
                                                
                                                # Create meshgrid for 3D surface
                                                time_mesh, shell_mesh = np.meshgrid(range(len(timesteps)), range(self.n_shells), indexing='ij')
                                                
                                                # Create 3D surface plot
                                                surf = ax.plot_surface(time_mesh, shell_mesh, maneuver_array, 
                                                                      cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
                                                
                                                ax.set_xlabel('Time Step')
                                                ax.set_ylabel('Shell Index')
                                                ax.set_zlabel('Number of Maneuvers')
                                                ax.set_title(f'3D Maneuver Heatmap - Species {species_name}')
                                                
                                                # Add colorbar
                                                fig.colorbar(surf, shrink=0.5, aspect=5)
                                                
                                                # Save plot
                                                filename = f'{species_name}_maneuvers.png'
                                                filepath = os.path.join(maneuvers_dir, filename)
                                                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                                                plt.close()
                                                
                                                print(f"Saved 3D maneuver plot: {filename}")
                        
                        # Create combined maneuver plot
                        self._create_combined_maneuver_plot(plot_data, other_data, maneuvers_dir)
                        
                except Exception as e:
                        print(f"Error creating 3D maneuver plots: {e}")

        def _create_3d_collision_plots(self, plot_data, other_data):
                """
                Create 3D collision plots for each species over time.
                """
                # Create collisions folder
                collisions_dir = os.path.join(plot_data.path, "collisions")
                os.makedirs(collisions_dir, exist_ok=True)
                
                try:
                        # Get time steps
                        timesteps = sorted(other_data.keys(), key=int)
                        
                        # Create 3D plot for each species
                        for species_name in plot_data.species_names:
                                if species_name.startswith('S'):  # Only active species
                                        # Extract collision probability data over time
                                        collision_data = []
                                        for timestep in timesteps:
                                                if 'collision_probability_all_species' in other_data[timestep]:
                                                        collision_probs = other_data[timestep]['collision_probability_all_species']
                                                        if isinstance(collision_probs, dict) and species_name in collision_probs:
                                                                collision_data.append(collision_probs[species_name])
                                                        else:
                                                                collision_data.append([0] * self.n_shells)  # Default to zeros
                                                else:
                                                        collision_data.append([0] * self.n_shells)  # Default to zeros
                                        
                                        if collision_data:
                                                # Convert to numpy array
                                                collision_array = np.array(collision_data)
                                                
                                                # Create 3D plot
                                                fig = plt.figure(figsize=(12, 10))
                                                ax = fig.add_subplot(111, projection='3d')
                                                
                                                # Create meshgrid for 3D surface
                                                time_mesh, shell_mesh = np.meshgrid(range(len(timesteps)), range(self.n_shells), indexing='ij')
                                                
                                                # Create 3D surface plot
                                                surf = ax.plot_surface(time_mesh, shell_mesh, collision_array, 
                                                                      cmap='plasma', alpha=0.8, linewidth=0, antialiased=True)
                                                
                                                ax.set_xlabel('Time Step')
                                                ax.set_ylabel('Shell Index')
                                                ax.set_zlabel('Collision Probability')
                                                ax.set_title(f'3D Collision Probability Heatmap - Species {species_name}')
                                                
                                                # Add colorbar
                                                fig.colorbar(surf, shrink=0.5, aspect=5)
                                                
                                                # Save plot
                                                filename = f'{species_name}_collisions.png'
                                                filepath = os.path.join(collisions_dir, filename)
                                                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                                                plt.close()
                                                
                                                print(f"Saved 3D collision plot: {filename}")
                        
                        # Create combined collision plot
                        self._create_combined_collision_plot(plot_data, other_data, collisions_dir)
                        
                except Exception as e:
                        print(f"Error creating 3D collision plots: {e}")

        def _create_combined_maneuver_plot(self, plot_data, other_data, maneuvers_dir):
                """
                Create a combined 3D maneuver plot showing all species together.
                """
                try:
                        timesteps = sorted(other_data.keys(), key=int)
                        active_species = [sp for sp in plot_data.species_names if sp.startswith('S')]
                        
                        if not active_species:
                                return
                        
                        # Create subplot grid
                        n_species = len(active_species)
                        n_cols = min(3, n_species)
                        n_rows = (n_species + n_cols - 1) // n_cols
                        
                        fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
                        
                        for i, species_name in enumerate(active_species):
                                ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
                                
                                # Extract maneuver data for this species
                                maneuver_data = []
                                for timestep in timesteps:
                                        if 'maneuvers' in other_data[timestep] and other_data[timestep]['maneuvers'] is not None:
                                                if isinstance(other_data[timestep]['maneuvers'], dict) and species_name in other_data[timestep]['maneuvers']:
                                                        maneuver_data.append(other_data[timestep]['maneuvers'][species_name])
                                                else:
                                                        maneuver_data.append([0] * self.n_shells)
                                        else:
                                                maneuver_data.append([0] * self.n_shells)
                                
                                if maneuver_data:
                                        maneuver_array = np.array(maneuver_data)
                                        time_mesh, shell_mesh = np.meshgrid(range(len(timesteps)), range(self.n_shells), indexing='ij')
                                        
                                        surf = ax.plot_surface(time_mesh, shell_mesh, maneuver_array, 
                                                              cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
                                        
                                        ax.set_xlabel('Time Step')
                                        ax.set_ylabel('Shell Index')
                                        ax.set_zlabel('Maneuvers')
                                        ax.set_title(f'{species_name} Maneuvers')
                        
                        plt.tight_layout()
                        
                        # Save combined plot
                        filepath = os.path.join(maneuvers_dir, 'combined_maneuvers_3d.png')
                        plt.savefig(filepath, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"Saved combined maneuver plot: combined_maneuvers_3d.png")
                        
                except Exception as e:
                        print(f"Error creating combined maneuver plot: {e}")

        def _create_combined_collision_plot(self, plot_data, other_data, collisions_dir):
                """
                Create a combined 3D collision plot showing all species together.
                """
                try:
                        timesteps = sorted(other_data.keys(), key=int)
                        active_species = [sp for sp in plot_data.species_names if sp.startswith('S')]
                        
                        if not active_species:
                                return
                        
                        # Create subplot grid
                        n_species = len(active_species)
                        n_cols = min(3, n_species)
                        n_rows = (n_species + n_cols - 1) // n_cols
                        
                        fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
                        
                        for i, species_name in enumerate(active_species):
                                ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
                                
                                # Extract collision data for this species
                                collision_data = []
                                for timestep in timesteps:
                                        if 'collision_probability_all_species' in other_data[timestep]:
                                                collision_probs = other_data[timestep]['collision_probability_all_species']
                                                if isinstance(collision_probs, dict) and species_name in collision_probs:
                                                        collision_data.append(collision_probs[species_name])
                                                else:
                                                        collision_data.append([0] * self.n_shells)
                                        else:
                                                collision_data.append([0] * self.n_shells)
                                
                                if collision_data:
                                        collision_array = np.array(collision_data)
                                        time_mesh, shell_mesh = np.meshgrid(range(len(timesteps)), range(self.n_shells), indexing='ij')
                                        
                                        surf = ax.plot_surface(time_mesh, shell_mesh, collision_array, 
                                                              cmap='plasma', alpha=0.8, linewidth=0, antialiased=True)
                                        
                                        ax.set_xlabel('Time Step')
                                        ax.set_ylabel('Shell Index')
                                        ax.set_zlabel('Collision Probability')
                                        ax.set_title(f'{species_name} Collisions')
                        
                        plt.tight_layout()
                        
                        # Save combined plot
                        filepath = os.path.join(collisions_dir, 'combined_collisions_3d.png')
                        plt.savefig(filepath, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"Saved combined collision plot: combined_collisions_3d.png")
                        
                except Exception as e:
                        print(f"Error creating combined collision plot: {e}")

        def _create_economic_metrics_plots(self, plot_data, other_data, econ_params):
                """
                Create plots for cost and rate of return over time for each species.
                """
                # Create economic_metrics folder
                econ_metrics_dir = os.path.join(plot_data.path, "economic_metrics")
                os.makedirs(econ_metrics_dir, exist_ok=True)
                
                try:
                        # Get time steps
                        timesteps = sorted(other_data.keys(), key=int)
                        
                        # Create plots for each species
                        for species_name in plot_data.species_names:
                                if species_name.startswith('S'):  # Only active species
                                        # Extract rate of return data over time
                                        ror_data = []
                                        for timestep in timesteps:
                                                if 'ror' in other_data[timestep]:
                                                        ror_data.append(other_data[timestep]['ror'])
                                                else:
                                                        ror_data.append([0] * self.n_shells)  # Default to zeros
                                        
                                        # Extract cost data over time from other_data
                                        cost_data = []
                                        for timestep in timesteps:
                                                if 'cost' in other_data[timestep] and species_name in other_data[timestep]['cost']:
                                                        cost_data.append(other_data[timestep]['cost'][species_name])
                                                else:
                                                        cost_data.append([0] * self.n_shells)  # Default to zeros
                                        
                                        if ror_data and cost_data:
                                                # Convert to numpy arrays
                                                ror_array = np.array(ror_data)
                                                cost_array = np.array(cost_data)
                                                
                                                # Create figure with subplots
                                                fig = plt.figure(figsize=(15, 6))
                                                ax1 = fig.add_subplot(121, projection='3d')
                                                ax2 = fig.add_subplot(122, projection='3d')
                                                
                                                # Create year labels (assuming start year 2017)
                                                start_year = 2017
                                                year_labels = [start_year + i for i in range(len(timesteps))]
                                                
                                                # Plot 1: Rate of Return over time
                                                time_mesh, shell_mesh = np.meshgrid(range(len(timesteps)), range(self.n_shells), indexing='ij')
                                                surf1 = ax1.plot_surface(time_mesh, shell_mesh, ror_array, 
                                                                        cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
                                                ax1.set_xlabel('Year')
                                                ax1.set_ylabel('Altitude (km)')
                                                ax1.set_zlabel('Rate of Return')
                                                ax1.set_title(f'Rate of Return Over Time - {species_name}')
                                                # Set custom tick labels
                                                ax1.set_xticks(range(len(timesteps)))
                                                ax1.set_xticklabels(year_labels)
                                                ax1.set_yticks(range(self.n_shells))
                                                ax1.set_yticklabels([f'{alt:.0f}' for alt in self.HMid])
                                                fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
                                                
                                                # Plot 2: Cost over time (3D surface plot)
                                                surf2 = ax2.plot_surface(time_mesh, shell_mesh, cost_array, 
                                                                        cmap='plasma', alpha=0.8, linewidth=0, antialiased=True)
                                                ax2.set_xlabel('Year')
                                                ax2.set_ylabel('Altitude (km)')
                                                ax2.set_zlabel('Cost ($)')
                                                ax2.set_title(f'Cost Over Time - {species_name}')
                                                # Set custom tick labels
                                                ax2.set_xticks(range(len(timesteps)))
                                                ax2.set_xticklabels(year_labels)
                                                ax2.set_yticks(range(self.n_shells))
                                                ax2.set_yticklabels([f'{alt:.0f}' for alt in self.HMid])
                                                fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
                                                
                                                plt.tight_layout()
                                                
                                                # Save individual species plot
                                                filename = f'{species_name}_economic_metrics.png'
                                                filepath = os.path.join(econ_metrics_dir, filename)
                                                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                                                plt.close()
                                                
                                                print(f"Saved economic metrics plot: {filename}")
                        
                        # Create combined economic metrics plot
                        self._create_combined_economic_metrics_plot(plot_data, other_data, econ_params, econ_metrics_dir)
                        
                except Exception as e:
                        print(f"Error creating economic metrics plots: {e}")

        def _create_combined_economic_metrics_plot(self, plot_data, other_data, econ_params, econ_metrics_dir):
                """
                Create a combined plot showing cost and rate of return for all species.
                """
                try:
                        timesteps = sorted(other_data.keys(), key=int)
                        active_species = [sp for sp in plot_data.species_names if sp.startswith('S')]
                        
                        if not active_species:
                                return
                        
                        # Create figure with subplots
                        fig = plt.figure(figsize=(20, 12))
                        
                        # Create subplot grid: 2 rows (RoR and Cost) x number of species columns
                        n_species = len(active_species)
                        
                        # Create year labels (assuming start year 2017)
                        start_year = 2017
                        year_labels = [start_year + i for i in range(len(timesteps))]
                        
                        # Row 1: Rate of Return plots
                        for i, species_name in enumerate(active_species):
                                ax = fig.add_subplot(2, n_species, i + 1, projection='3d')
                                
                                # Extract rate of return data for this species
                                ror_data = []
                                for timestep in timesteps:
                                        if 'ror' in other_data[timestep]:
                                                ror_data.append(other_data[timestep]['ror'])
                                        else:
                                                ror_data.append([0] * self.n_shells)
                                
                                if ror_data:
                                        ror_array = np.array(ror_data)
                                        time_mesh, shell_mesh = np.meshgrid(range(len(timesteps)), range(self.n_shells), indexing='ij')
                                        
                                        surf = ax.plot_surface(time_mesh, shell_mesh, ror_array, 
                                                              cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
                                        
                                        ax.set_xlabel('Year')
                                        ax.set_ylabel('Altitude (km)')
                                        ax.set_zlabel('Rate of Return')
                                        ax.set_title(f'{species_name} - Rate of Return')
                                        # Set custom tick labels
                                        ax.set_xticks(range(len(timesteps)))
                                        ax.set_xticklabels(year_labels)
                                        ax.set_yticks(range(self.n_shells))
                                        ax.set_yticklabels([f'{alt:.0f}' for alt in self.HMid])
                        
                        # Row 2: Cost plots
                        for i, species_name in enumerate(active_species):
                                ax = fig.add_subplot(2, n_species, n_species + i + 1, projection='3d')
                                
                                # Extract cost data over time for this species
                                cost_data = []
                                for timestep in timesteps:
                                        if 'cost' in other_data[timestep] and species_name in other_data[timestep]['cost']:
                                                cost_data.append(other_data[timestep]['cost'][species_name])
                                        else:
                                                cost_data.append([0] * self.n_shells)
                                
                                if cost_data:
                                        cost_array = np.array(cost_data)
                                        time_mesh, shell_mesh = np.meshgrid(range(len(timesteps)), range(self.n_shells), indexing='ij')
                                        
                                        surf = ax.plot_surface(time_mesh, shell_mesh, cost_array, 
                                                              cmap='plasma', alpha=0.8, linewidth=0, antialiased=True)
                                        
                                        ax.set_xlabel('Year')
                                        ax.set_ylabel('Altitude (km)')
                                        ax.set_zlabel('Cost ($)')
                                        ax.set_title(f'{species_name} - Cost Over Time')
                                        # Set custom tick labels
                                        ax.set_xticks(range(len(timesteps)))
                                        ax.set_xticklabels(year_labels)
                                        ax.set_yticks(range(self.n_shells))
                                        ax.set_yticklabels([f'{alt:.0f}' for alt in self.HMid])
                        
                        plt.tight_layout()
                        
                        # Save combined plot
                        filepath = os.path.join(econ_metrics_dir, 'combined_economic_metrics.png')
                        plt.savefig(filepath, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"Saved combined economic metrics plot: combined_economic_metrics.png")
                        
                except Exception as e:
                        print(f"Error creating combined economic metrics plot: {e}")

        def comparison_total_bond_revenue_vs_time(self, plot_data_lists, other_data_lists):
                """
                Plots the total bond revenue collected over time for each scenario.
                Revenue = (bond amount) * (number of non-compliant satellites).
                Also saves cumulative revenue data to CSV.
                """
                import matplotlib.pyplot as plt
                import numpy as np
                import os
                import re
                import pandas as pd
                
                plt.figure(figsize=(10, 6))
                comparison_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(comparison_folder, exist_ok=True)

                def format_bond_label(bond_amount):
                        """Format bond amount as $100k, $500k, etc."""
                        if bond_amount == 0:
                                return "$0"
                        elif bond_amount < 1000:
                                return f"${int(bond_amount)}"
                        elif bond_amount < 1000000:
                                # Format as $100k, $500k, etc.
                                return f"${int(bond_amount / 1000)}k"
                        else:
                                # Format as $1M, $2M, etc.
                                return f"${bond_amount / 1000000:.1f}M"

                # Store data for CSV export
                csv_data = []

                for plot_data, other_data in zip(plot_data_lists, other_data_lists):
                        scenario_label = getattr(plot_data, 'scenario', 'Unnamed Scenario')
                        
                        # Get the loaded econ_params dictionary, e.g., { "SA": {...}, "SB": {...}, ... }
                        econ_params_dict = plot_data.econ_params
                        
                        # Find the S-species to check for the bond amount
                        # We use plot_data.data.keys() to get the list of species
                        s_species_names = [sp for sp in plot_data.data.keys() if sp.startswith("S")]
                        
                        # Get bond amount from the first S-species. Default to 0 if no bond.
                        bond_amount = 0.0
                        disposal_time = None
                        if s_species_names and s_species_names[0] in econ_params_dict:
                                bond_amount = econ_params_dict[s_species_names[0]].get("bond") or 0.0
                                disposal_time = econ_params_dict[s_species_names[0]].get("disposal_time")
                        
                        # If disposal_time not found in econ_params, try to extract from scenario name
                        if disposal_time is None:
                                # Check scenario name for 5yr or 25yr pattern
                                if "25yr" in scenario_label:
                                        disposal_time = 25
                                elif "5yr" in scenario_label:
                                        disposal_time = 5
                                else:
                                        disposal_time = 5  # Default to 5 years
                        
                        # Format bond amount for legend
                        bond_label = format_bond_label(bond_amount)
                        
                        # Get sorted timesteps (e.g., '1', '2', ..., '9')
                        timesteps = sorted(other_data.keys(), key=int)
                        
                        # 0 for the initial year (year 0)
                        bond_revenues = [0.0] 
                        
                        for ts in timesteps:
                                # Get the non-compliance dictionary {species: count}
                                non_compliance_data = other_data[ts].get("non_compliance", {})
                                # Sum the counts of non-compliant sats across all species
                                total_non_compliant = sum(non_compliance_data.values())
                                bond_revenues.append(total_non_compliant * bond_amount)

                        # Calculate cumulative revenue (sum of all years)
                        cumulative_revenue = sum(bond_revenues)
                        
                        # Store data for CSV
                        csv_data.append({
                                'bond_amount': bond_amount,
                                'bond_label': bond_label,
                                'disposal_time_years': disposal_time,
                                'cumulative_revenue': cumulative_revenue
                        })

                        time_axis = range(len(bond_revenues))
                        plt.plot(time_axis, bond_revenues, label=bond_label, linewidth=2, marker='o', markersize=4)

                plt.xlabel("Year", fontsize=16, fontweight="black")
                plt.ylabel("Total Bond Revenue ($)", fontsize=16, fontweight="black")
                
                # Make axis ticks bold
                ax = plt.gca()
                ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)
                for label in ax.get_xticklabels():
                        label.set_fontweight('bold')
                for label in ax.get_yticklabels():
                        label.set_fontweight('bold')
                
                plt.legend(fontsize=13, frameon=True, loc="upper left")
                if plt.gca().get_legend():
                        for text in plt.gca().get_legend().get_texts():
                                text.set_fontweight("black")
                plt.grid(True)
                plt.tight_layout()

                file_path = os.path.join(comparison_folder, "total_bond_revenue_vs_time.png")
                plt.savefig(file_path, dpi=300)
                plt.close()
                print(f"Bond revenue plot saved to {file_path}")
                
                # Save cumulative revenue data to CSV
                if csv_data:
                        df = pd.DataFrame(csv_data)
                        # Sort by bond amount, then by disposal time
                        df = df.sort_values(['bond_amount', 'disposal_time_years'])
                        # Reorder columns
                        df = df[['bond_amount', 'bond_label', 'disposal_time_years', 'cumulative_revenue']]
                        csv_path = os.path.join(comparison_folder, "cumulative_bond_revenue.csv")
                        df.to_csv(csv_path, index=False)
                        print(f"Cumulative bond revenue CSV saved to {csv_path}")