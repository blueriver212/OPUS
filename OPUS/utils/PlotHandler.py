import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import math

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
                                plot_method = attr
                                plot_method(plot_data_lists, other_data_lists)

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

        def comparison_total_species_count(self, plot_data_lists, other_data_lists=None):
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
                        for i, (scenario_name, counts) in enumerate(scenario_data.items()):
                                x_axis = range(len(counts)) + np.ones(len(counts))
                                if i <= 9:
                                        ax.plot(x_axis, counts, label=scenario_name, marker='o')
                                elif (i > 9) and (i <= 19):
                                        ax.plot(x_axis, counts, label=scenario_name, marker='X')
                                else:
                                        ax.plot(x_axis, counts, label=scenario_name, marker='>')

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

        # # def comparison_total_species_count(self):
        # #         """
        # #         Creates a comparison plot of total species count over time.
        # #         Each species is plotted in its own subplot, comparing across all scenarios.
        # #         """

        # #         # Create a "comparisons" folder under the main simulation folder
        # #         comparison_folder = os.path.join(self.simulation_folder, "comparisons")
        # #         os.makedirs(comparison_folder, exist_ok=True)

        # #         # Dictionary to store time series data for each species across scenarios
        # #         species_totals = {}

        # #         # Loop over each scenario to extract data
        # #         for scenario in self.scenario_files:
        # #                 scenario_folder = os.path.join(self.simulation_folder, scenario)
                        
        # #                 if not os.path.exists(scenario_folder):
        # #                         print(f"Warning: Cannot find {scenario_folder}; skipping.")
        # #                         continue

        # #                 # Build PlotData to get the dictionary of species->data arrays
        # #                 plot_data = PlotData(scenario, scenario_folder, self.MOCAT)
        # #                 data_dict = plot_data.data  # {species: np.array(time, shells), ...}

        # #                 for species, species_data in data_dict.items():
        # #                         # Sum across shells to get a total count per time step
        # #                         total_species_count = np.sum(species_data, axis=1)  # shape: (time,)

        # #                         # Store data per species
        # #                         if species not in species_totals:
        # #                                 species_totals[species] = {}
                                
        # #                         species_totals[species][scenario] = total_species_count

        # #         # Count how many species we have
        # #         num_species = len(species_totals)

        # #         # If multiple species, create subplots in a grid
        # #         num_cols = 2
        # #         num_rows = math.ceil(num_species / num_cols)

        # #         fig, axes = plt.subplots(num_rows, num_cols,
        # #                                 figsize=(12, 6 * num_rows),
        # #                                 sharex=True)
        # #         # Flatten axes for easy iteration
        # #         axes = np.array(axes).flatten()

        # #         for idx, (species, scenario_data) in enumerate(species_totals.items()):
        # #                 ax = axes[idx]
        # #                 for scenario, counts in scenario_data.items():
        # #                         ax.plot(counts, label=scenario, marker='o')

        # #                 ax.set_title(f"Total Count across all shells for Species: {species}")
        # #                 ax.set_xlabel("Year")
        # #                 ax.set_ylabel("Total Count")
        # #                 ax.legend()
        # #                 ax.grid(True)

        # #         # Hide any leftover empty subplots (if #species not a multiple of num_cols)
        # #         for extra_ax in axes[num_species:]:
        # #                 extra_ax.set_visible(False)

        # #         plt.tight_layout()

        # #         # Save the figure
        # #         out_path = os.path.join(comparison_folder, "comparison_species_count.png")
        # #         plt.savefig(out_path, dpi=300)
        # #         plt.close()
        # #         print(f"Comparison plot saved to {out_path}")

        # def comparison_UMPY(self, plot_data_lists, other_data_lists):
        #         """
        #         Create a comparison plot of total UMPY over time for multiple scenarios.
        #         Each scenario is plotted on the same figure with a label derived from 
        #         its scenario name.
        #         """
        #         comparison_folder = os.path.join(self.simulation_folder, "comparisons")
        #         os.makedirs(comparison_folder, exist_ok=True)

        #         # Create a single figure for all scenarios
        #         plt.figure(figsize=(8, 5))

        #         # Loop through each plot_data and other_data pair
        #         for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
        #                 # 1) Sort the timesteps
        #                 timesteps = sorted(other_data.keys(), key=int)
        #                 umpy_sums = []

        #                 # 2) Sum the 'umpy' values for each timestep
        #                 for ts in timesteps:
        #                         umpy_list = other_data[ts]["umpy"]  # This is assumed to be a list of floats
        #                         total_umpy = np.sum(umpy_list)
        #                         umpy_sums.append(total_umpy)

        #                 # Here we assume `plot_data` has an attribute storing the scenario name.
        #                 # Adjust this to match your actual code if the attribute differs.
        #                 scenario_label = getattr(plot_data, 'scenario', f"Scenario {i+1}")

        #                 # 3) Plot each scenario on the same figure
        #                 plt.plot(
        #                 timesteps,
        #                 umpy_sums,
        #                 marker='o',
        #                 label=scenario_label
        #                 )

        #         # 4) Labels, legend, and layout
        #         plt.xlabel("Year (timestep)")
        #         plt.ylabel("UMPY (kg/year)")
        #         plt.title("UMPY Evolution Over Time (All Scenarios)")
        #         plt.legend()
        #         plt.tight_layout()

        #         # 5) Save the figure using the first plot_data's path 
        #         out_path = os.path.join(comparison_folder, "umpy_over_time.png")
        #         plt.savefig(out_path, dpi=300, bbox_inches="tight")
        #         plt.close()
        #         print(f"Comparison UMPY plot saved to {out_path}")

        # sammie addition
        def comparison_welfare_vs_tax(self, plot_data_lists, other_data_lists):

                tax_rate = {}

                for scenario_name in self.scenario_files:
                        file = open(f'./Results/{self.simulation_name}/{scenario_name}/econ_params_{scenario_name}.json')
                        econ_params = json.load(file)
                        for k, v in econ_params.items():
                                if k == "tax":
                                        tax = v
                        if scenario_name not in tax_rate:
                                tax_rate[scenario_name] = None
                        tax_rate[scenario_name] = tax

                
                scatter_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(scatter_folder, exist_ok=True)
                scatter_file_path = os.path.join(scatter_folder, "scatter_final_welfare_vs_tax.png")
                
                tax_values = []
                final_welfare_values = []
                labels = []
                
                # Loop over each scenario's data
                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        # --- Calculate final UMPY value ---
                        timesteps = sorted(other_data.keys(), key=int)
                        last_timestep = timesteps[-1]
                        # Sum the "umpy" list at the final timestep
                        final_welfare = other_data[last_timestep]["welfare"]
                        
                        final_welfare_values.append(final_welfare)
                        scenario_label = getattr(plot_data, "scenario", f"Scenario {i+1}")
                        for k, v in tax_rate.items():
                                if k == scenario_label:
                                        tax_values.append(v)
                        labels.append(scenario_label)
                
                # --- Create scatter plot ---
                plt.figure(figsize=(8, 6))
                plt.scatter(tax_values, final_welfare_values, marker='o')
                
                # Annotate each point with its scenario label
                for x, y, label in zip(tax_values, final_welfare_values, labels):
                        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), ha="left")
                        
                plt.xlabel("Tax Rate")
                plt.ylabel("Final Welfare ($)")
                plt.title("Tax Rate vs Final Welfare by Scenario")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(scatter_file_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Scatter plot saved to {scatter_file_path}")

        # sammie addition
        def comparison_umpy_vs_tax(self, plot_data_lists, other_data_lists):
                tax_rate = {}

                for scenario_name in self.scenario_files:
                        file = open(f'./Results/{self.simulation_name}/{scenario_name}/econ_params_{scenario_name}.json')
                        econ_params = json.load(file)
                        for k, v in econ_params.items():
                                if k == "tax":
                                        tax = v
                        if scenario_name not in tax_rate:
                                tax_rate[scenario_name] = None
                        tax_rate[scenario_name] = tax

                
                scatter_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(scatter_folder, exist_ok=True)
                scatter_file_path = os.path.join(scatter_folder, "scatter_final_umpy_vs_tax.png")
                
                tax_values = []
                final_umpy_values = []
                labels = []
                
                # Loop over each scenario's data
                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        # --- Calculate final UMPY value ---
                        timesteps = sorted(other_data.keys(), key=int)
                        last_timestep = timesteps[-1]
                        # Sum the "umpy" list at the final timestep
                        final_umpy = np.sum(other_data[last_timestep]["umpy"])
                        final_umpy_values.append(final_umpy)
                        scenario_label = getattr(plot_data, "scenario", f"Scenario {i+1}")
                        for k, v in tax_rate.items():
                                if k == scenario_label:
                                        tax_values.append(v)
                        labels.append(scenario_label)
                
                # --- Create scatter plot ---
                plt.figure(figsize=(8, 6))
                plt.scatter(tax_values, final_umpy_values, marker='o')
                
                # Annotate each point with its scenario label
                for x, y, label in zip(tax_values, final_umpy_values, labels):
                        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), ha="left")
                        
                plt.xlabel("Tax Rate")
                plt.ylabel("Final UMPY (kg/year)")
                plt.title("Tax Rate vs Final UMPY by Scenario")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(scatter_file_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Scatter plot saved to {scatter_file_path}")

        def comparison_time_welfare(self, plot_data_lists, other_data_lists):
                comparison_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(comparison_folder, exist_ok=True)
                
                welfare_dict = {}
                
                # Loop over each scenario's data
                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        # We assume `plot_data.scenario` holds the scenario name
                        # If it doesn't, change the attribute name or use a default label.
                        scenario_name = getattr(plot_data, "scenario", f"Scenario {i+1}")

                        # Retrieve the dictionary of species -> data arrays
                        # e.g., {species: np.array(time, shells), ...

                        for j, year in enumerate(other_data):
                                welfare = other_data[year]['welfare']
                                if scenario_name not in welfare_dict:
                                        welfare_dict[scenario_name] = np.zeros(len(other_data))
                                welfare_dict[scenario_name][j] = welfare
                counter = 0
                labels = []
                for idx, scenario_name in enumerate(welfare_dict):
                        welfare_list = welfare_dict[scenario_name]
                        x_axis = range(len(welfare_list)) + np.ones(len(welfare_list))
                        if idx <= 9:
                                plt.plot(x_axis, welfare_list, label=scenario_name, marker='o')
                                counter = counter + 1
                        elif (idx > 9) and (idx <= 19):
                                plt.plot(x_axis, welfare_list, label=scenario_name, marker='X')
                                counter = counter + 1
                        else:
                                counter = counter + 1
                                plt.plot(x_axis, welfare_list, label=scenario_name, marker='>')
                        labels.append(scenario_name)
                        
                plt.title(f"Welfare Over Time")
                plt.xlabel("Time Steps (Years)")
                plt.ylabel("Welfare ($)")
                plt.legend()
                plt.grid(True)

                out_path = os.path.join(comparison_folder, "welfare_comparison.png")
                plt.savefig(out_path, dpi=300)

                print(f"Comparison launch plot saved to {out_path}")
                print("Counter: ", counter)

        def comparison_final_umpy_vs_total_count(self, plot_data_lists, other_data_lists):
                """
                Create and save a scatter plot where each scenario is represented by a point.
                The X axis is the final UMPY value (kg/year) and the Y axis is the final total count of objects.
                
                Parameters
                ----------
                plot_data_lists : list
                        A list of PlotData objects containing species count data in the 'data' attribute.
                other_data_lists : list
                        A list of dictionaries (one per scenario) that include timesteps with an "umpy" key.
                """
                # Create folder for comparison plots
                scatter_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(scatter_folder, exist_ok=True)
                scatter_file_path = os.path.join(scatter_folder, "scatter_final_umpy_vs_total_count.png")
                
                final_umpy_values = []
                final_total_counts = []
                labels = []
                
                # Loop over each scenario's data
                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        # --- Calculate final UMPY value ---
                        timesteps = sorted(other_data.keys(), key=int)
                        last_timestep = timesteps[-1]
                        # Sum the "umpy" list at the final timestep
                        final_umpy = np.sum(other_data[last_timestep]["umpy"])
                        
                        # --- Calculate final total count of objects ---
                        # Assumes plot_data.data is a dictionary: {species: 2D array with shape (time, shells), ...}
                        total_count = 0
                        for sp, data in plot_data.data.items():
                                data_arr = np.array(data)  # ensure it's a NumPy array
                                # Sum the counts for the final time step (assumed to be the first dimension)
                                final_count_sp = np.sum(data_arr[-1, :])
                                total_count += final_count_sp
                        
                        final_umpy_values.append(final_umpy)
                        final_total_counts.append(total_count)
                        scenario_label = getattr(plot_data, "scenario", f"Scenario {i+1}")
                        labels.append(scenario_label)
                
                # --- Create scatter plot ---
                plt.figure(figsize=(8, 6))
                plt.scatter(final_umpy_values, final_total_counts, marker='o')
                
                # Annotate each point with its scenario label
                for x, y, label in zip(final_umpy_values, final_total_counts, labels):
                        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), ha="left")
                        
                plt.xlabel("Final UMPY (kg/year)")
                plt.ylabel("Final Total Count of Objects")
                plt.title("Final UMPY vs Final Total Count by Scenario")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(scatter_file_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Scatter plot saved to {scatter_file_path}")

        # sammie addition:
        def comparison_umpy_vs_welfare(self, plot_data_lists, other_data_lists):
                # Create folder for comparison plots
                scatter_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(scatter_folder, exist_ok=True)
                scatter_file_path = os.path.join(scatter_folder, "scatter_final_umpy_vs_welfare.png")
                
                final_umpy_values = []
                final_welfare_values = []
                labels = []
                
                # Loop over each scenario's data
                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        # --- Calculate final UMPY value ---
                        timesteps = sorted(other_data.keys(), key=int)
                        last_timestep = timesteps[-1]
                        # Sum the "umpy" list at the final timestep
                        final_umpy = np.sum(other_data[last_timestep]["umpy"])
                        final_welfare = other_data[last_timestep]["welfare"]

                        final_umpy_values.append(final_umpy)
                        final_welfare_values.append(final_welfare)
                        scenario_label = getattr(plot_data, "scenario", f"Scenario {i+1}")
                        labels.append(scenario_label)
                
                # --- Create scatter plot ---
                plt.figure(figsize=(8, 6))
                plt.scatter(final_umpy_values, final_welfare_values, marker='o')
                
                # Annotate each point with its scenario label
                for x, y, label in zip(final_umpy_values, final_welfare_values, labels):
                        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), ha="left")
                        
                plt.xlabel("Final UMPY (kg/year)")
                plt.ylabel("Final Welfare ($)")
                plt.title("Final UMPY vs Final Welfare by Scenario")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(scatter_file_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Scatter plot saved to {scatter_file_path}")

        def comparison_umpy_vs_count_and_collision(self, plot_data_lists, other_data_lists):
                """
                Create a side-by-side scatter plot with two subplots:
                - Left subplot: Final UMPY (x-axis) vs. Final Total Count of Objects (y-axis)
                - Right subplot: Final UMPY (x-axis) vs. Final Collision Probability (y-axis)
                
                For each scenario:
                - Final UMPY is computed as the sum of the "umpy" array at the final timestep.
                - Final Total Count is computed by summing, for each species in plot_data.data,
                        the counts from the last row (final timestep) across all shells.
                - Final Collision Probability is computed as the sum of the 
                        "collision_probability_all_species" array at the final timestep.
                """
                # Create a folder for saving the plot.
                scatter_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(scatter_folder, exist_ok=True)
                file_path = os.path.join(scatter_folder, "final_umpy_vs_count_and_collision.png")
                
                # Prepare lists to store values and labels for each scenario.
                final_umpy_vals = []
                final_total_counts = []
                final_collision_probs = []
                labels = []
                
                # Loop over scenarios.
                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        # Get scenario label (or default if not available)
                        scenario_label = getattr(plot_data, 'scenario', f"Scenario {i+1}")
                        
                        # Sort timesteps and get the final one.
                        timesteps = sorted(other_data.keys(), key=int)
                        last_timestep = timesteps[-1]
                        
                        # Compute final UMPY (x-axis value)
                        final_umpy = np.sum(other_data[last_timestep]["umpy"])
                        
                        # Compute final total count (sum over species from plot_data.data)
                        total_count = 0
                        for sp, data in plot_data.data.items():
                        # Assume data is an array-like with shape (time, shells)
                                arr = np.array(data)
                                total_count += np.sum(arr[-1, :])
                        
                        # Compute final collision probability by summing the provided array.
                        final_collision = np.sum(other_data[last_timestep]["collision_probability"])
                        
                        final_umpy_vals.append(final_umpy)
                        final_total_counts.append(total_count)
                        final_collision_probs.append(final_collision)
                        labels.append(scenario_label)
                
                # Create a figure with two subplots side by side.
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # Left subplot: Final UMPY vs. Final Total Count
                axes[0].scatter(final_umpy_vals, final_total_counts, s=100, zorder=3)
                for x, y, label in zip(final_umpy_vals, final_total_counts, labels):
                        axes[0].annotate(label, (x, y), textcoords="offset points", xytext=(5, 5))
                axes[0].set_xlabel("Final UMPY (kg/year)")
                axes[0].set_ylabel("Final Total Count of Objects")
                axes[0].set_title("Final UMPY vs. Total Count")
                axes[0].grid(True, zorder=0)
                
                # Right subplot: Final UMPY vs. Final Collision Probability
                axes[1].scatter(final_umpy_vals, final_collision_probs, s=100, zorder=3)
                for x, y, label in zip(final_umpy_vals, final_collision_probs, labels):
                        axes[1].annotate(label, (x, y), textcoords="offset points", xytext=(5, 5))
                axes[1].set_xlabel("Final UMPY (kg/year)")
                axes[1].set_ylabel("Final Collision Probability")
                axes[1].set_title("Final UMPY vs. Collision Probability")
                axes[1].grid(True, zorder=0)
                
                plt.tight_layout()
                plt.savefig(file_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Final UMPY vs. Count and Collision Probability scatter plots saved to {file_path}")

        # sammie addition:
        def comparison_total_launch(self, plot_data_lists, other_data_lists):
                """
                Creates a comparison plot of total species count over time.
                Each species is plotted in its own subplot, comparing across all scenarios.
                """

                # Create a "comparisons" folder under the main simulation folder
                comparison_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(comparison_folder, exist_ok=True)

                # Dictionary to store time series data for each species across scenarios
                # Structure: species_totals[species][scenario_name] = np.array of total counts over time
                launch_totals = {}

                # Loop over each PlotData to extract data
                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        # We assume `plot_data.scenario` holds the scenario name
                        # If it doesn't, change the attribute name or use a default label.
                        scenario_name = getattr(plot_data, "scenario", f"Scenario {i+1}")

                        # Retrieve the dictionary of species -> data arrays
                        # e.g., {species: np.array(time, shells), ...

                        
                        for j, year in enumerate(other_data):
                                launches = other_data[year]['launch_rate']
                                if scenario_name not in launch_totals:
                                        launch_totals[scenario_name] = np.zeros(len(other_data))
                                launch_totals[scenario_name][j] = np.sum(launches)
                
                for idx, scenario_name in enumerate(launch_totals):
                        launches = launch_totals[scenario_name]
                        x_axis = range(len(launches)) + np.ones(len(launches))
                        if idx <= 9:
                                plt.plot(x_axis, launches, label=scenario_name, marker='o')
                        elif (idx > 9) and (idx <= 19):
                                plt.plot(x_axis, launches, label=scenario_name, marker='X')
                        else:
                                plt.plot(x_axis, launches, label=scenario_name, marker='>')
                        
                plt.title(f"Total Launches across all shells")
                plt.xlabel("Time Steps (Years)")
                plt.ylabel("Total Launches")
                plt.legend()
                plt.grid(True)

                out_path = os.path.join(comparison_folder, "total_launches_comparison.png")
                plt.savefig(out_path, dpi=300)
                plt.close()

                print(f"Comparison launch plot saved to {out_path}")

        # sammie addition:
        def comparison_initial_conditions(self, plot_data_lists, other_data_lists):
                """
                Creates a comparison plot of total species count over time subtracting the initial amounts.
                Each species is plotted in its own subplot, comparing across all scenarios.
                """

                # Create a "comparisons" folder under the main simulation folder
                comparison_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(comparison_folder, exist_ok=True)

                # Dictionary to store time series data for each species across scenarios
                # Structure: species_totals[species][scenario_name] = np.array of total counts over time
                species_totals = {}
                IC = {}

                # Loop over each PlotData to extract data
                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        # We assume `plot_data.scenario` holds the scenario name
                        # If it doesn't, change the attribute name or use a default label.
                        scenario_name = getattr(plot_data, "scenario", f"Scenario {i+1}")
                        x0 = other_data["1"]["ICs"]
                        for idx, sp in enumerate(self.MOCAT.scenario_properties.species_names):
                                if sp not in IC:
                                        IC[sp] = np.zeros(self.n_shells)
                                IC[sp] = np.sum(x0[idx * self.n_shells:(idx + 1) * self.n_shells])


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
                                species_totals[species][scenario_name] = total_species_count - IC[species]

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
                        for i, (scenario_name, counts) in enumerate(scenario_data.items()):
                                x_axis = range(len(counts)) + np.ones(len(counts))
                                if i <= 9:
                                        ax.plot(x_axis, counts, label=scenario_name, marker='o')
                                elif (i > 9) and (i <= 19):
                                        ax.plot(x_axis, counts, label=scenario_name, marker='X')
                                else:
                                        ax.plot(x_axis, counts, label=scenario_name, marker='>')

                        ax.set_title(f"Change in Total Count across all shells: {species}")
                        ax.set_xlabel("Time Steps (or Years)")
                        ax.set_ylabel("Change in Total Count")
                        ax.legend()
                        ax.grid(True)

                # Hide any leftover empty subplots (if #species < num_rows * num_cols)
                for extra_ax in axes[num_species:]:
                        extra_ax.set_visible(False)

                plt.tight_layout()

                # Save the figure
                out_path = os.path.join(comparison_folder, "comparison_IC_difference.png")
                plt.savefig(out_path, dpi=300)
                plt.close()

                print(f"Comparison plot saved to {out_path}")

        # sammie addition
        def comparison_umpy_over_time(self, plot_data_lists, other_data_lists):
                # Create a "comparisons" folder under the main simulation folder
                comparison_folder = os.path.join(self.simulation_folder, "comparisons")
                os.makedirs(comparison_folder, exist_ok=True)

                # Dictionary to store time series data for each species across scenarios
                # Structure: species_totals[species][scenario_name] = np.array of total counts over time
                umpy_dict = {}


                for i, (plot_data, other_data) in enumerate(zip(plot_data_lists, other_data_lists)):
                        # We assume `plot_data.scenario` holds the scenario name
                        # If it doesn't, change the attribute name or use a default label.
                        scenario_name = getattr(plot_data, "scenario", f"Scenario {i+1}")

                        # Retrieve the dictionary of species -> data arrays
                        # e.g., {species: np.array(time, shells), ...

                        
                        for j, year in enumerate(other_data):
                                umpy = other_data[year]['umpy']
                                if scenario_name not in umpy_dict:
                                        umpy_dict[scenario_name] = np.zeros(len(other_data))
                                umpy_dict[scenario_name][j] = np.sum(umpy)
                
                for idx, scenario_name in enumerate(umpy_dict):
                        umpy = umpy_dict[scenario_name]
                        x_axis = range(len(umpy)) + np.ones(len(umpy))
                        if idx <= 9:
                                plt.plot(x_axis, umpy, label=scenario_name, marker='o')
                        elif (idx > 9) and (idx <= 19):
                                plt.plot(x_axis, umpy, label=scenario_name, marker='X')
                        else:
                                plt.plot(x_axis, umpy, label=scenario_name, marker='>')
                # 1) Sort the timesteps and prepare arrays

                plt.figure(figsize=(8, 5))

                # 4) Labels and title
                plt.xlabel("Year (timestep)")
                plt.ylabel("UMPY (kg/year)")
                plt.title("UMPY Evolution Over Time")
                plt.legend()
                plt.tight_layout()

                # 5) Save the figure
                save_path = os.path.join(plot_data.path, "umpy_comparison.png")
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()

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

        def ror_cp_and_launch_rate(self, plot_data, other_data):
                """
                Generate and save a combined plot for time evolution of different parameters (RoR, Collision Probability, Launch Rate).
                """
                # Extract keys (timesteps) and sort
                timesteps = sorted(other_data.keys(), key=int)

                # Get number of altitude shells (assuming all timesteps have the same length)
                num_altitude_shells = len(other_data[timesteps[0]]["ror"])

                # Prepare the figure
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Color map for time evolution
                colors = cm.viridis(np.linspace(0, 1, len(timesteps)))

                # Static Plot
                for idx, timestep in enumerate(timesteps):
                        ror = other_data[timestep]["ror"]
                        collision_prob = other_data[timestep]["collision_probability"]
                        launch_rate = other_data[timestep]["launch_rate"]

                        axes[0].plot(range(num_altitude_shells), ror, color=colors[idx], label=f"Year {timestep}")
                        axes[1].plot(range(num_altitude_shells), collision_prob, color=colors[idx])
                        axes[2].plot(range(num_altitude_shells), launch_rate, color=colors[idx])

                axes[0].set_title("Rate of Return (RoR)")
                axes[1].set_title("Collision Probability")
                axes[2].set_title("Launch Rate")

                # Set labels and ticks
                for ax in axes:
                        ax.set_xlabel("Shell - Mid Altitude (km)")
                        ax.set_ylabel("Value")
                        ax.set_xticklabels(self.HMid)

                # Add a legend
                fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)

                # Tight layout
                plt.tight_layout()

                # Save the combined plot
                combined_file_path = os.path.join(plot_data.path, "combined_time_evolution.png")
                plt.savefig(combined_file_path, dpi=300, bbox_inches='tight')
                plt.close()

        def ror_cp_and_launch_rate_gif(self, plot_data, other_data):
                """
                Generate and save an animated plot for the time evolution of different parameters (RoR, Collision Probability, Launch Rate).
                """
                # Extract keys (timesteps) and sort
                timesteps = sorted(other_data.keys(), key=int)

                # Get number of altitude shells (assuming all timesteps have the same length)
                num_altitude_shells = len(other_data[timesteps[0]]["ror"])

                # Determine global min/max for each metric across all timesteps
                ror_values = [val for timestep in timesteps for val in other_data[timestep]["ror"]]
                collision_values = [val for timestep in timesteps for val in other_data[timestep]["collision_probability"]]
                launch_values = [val for timestep in timesteps for val in other_data[timestep]["launch_rate"]]

                ror_min, ror_max = min(ror_values), max(ror_values)
                collision_min, collision_max = min(collision_values), max(collision_values)
                launch_min, launch_max = min(launch_values), max(launch_values)

                # Create the figure and axes
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                def update(frame):
                        timestep = timesteps[frame]
                        ror = other_data[timestep]["ror"]
                        collision_prob = other_data[timestep]["collision_probability"]
                        launch_rate = other_data[timestep]["launch_rate"]

                        for ax in axes:
                                ax.clear()

                                # Plot each metric with fixed y-axis limits
                                axes[0].plot(range(num_altitude_shells), ror, color='b')
                                axes[0].set_ylim(ror_min, ror_max)
                                axes[0].set_title(f"Rate of Return (RoR) - Year {timestep}")
                                axes[0].set_xlabel("Shell - Mid Altitude (km)")
                                axes[0].set_ylabel("RoR")
                                axes[0].set_xticks(range(len(self.HMid)))  # Ensure correct number of ticks
                                axes[0].set_xticklabels(self.HMid)

                                axes[1].plot(range(num_altitude_shells), collision_prob, color='r')
                                axes[1].set_ylim(collision_min, collision_max)
                                axes[1].set_title(f"Collision Probability - Year {timestep}")
                                axes[1].set_xlabel("Shell - Mid Altitude (km)")
                                axes[1].set_ylabel("Collision Probability")
                                axes[1].set_xticks(range(len(self.HMid)))  # Ensure correct number of ticks
                                axes[1].set_xticklabels(self.HMid)

                                axes[2].plot(range(num_altitude_shells), launch_rate, color='g')
                                axes[2].set_ylim(launch_min, launch_max)
                                axes[2].set_title(f"Launch Rate - Year {timestep}")
                                axes[2].set_xlabel("Shell - Mid Altitude (km)")
                                axes[2].set_ylabel("Launch Rate")
                                axes[2].set_xticks(range(len(self.HMid)))  # Ensure correct number of ticks
                                axes[2].set_xticklabels(self.HMid)

                        plt.tight_layout()
                
                # Create the animation
                ani = animation.FuncAnimation(fig, update, frames=len(timesteps), repeat=True)

                # Save as GIF
                combined_file_path = os.path.join(plot_data.path, "space_metrics_evolution.gif")
                ani.save(combined_file_path, writer="pillow", fps=2)

                plt.close()

        def ror_cp_and_launch_rate(self, plot_data, other_data):
                """
                Generate and save a combined plot for time evolution of different parameters
                (RoR, Collision Probability, Launch Rate, Excess Returns).
                """
                # Extract keys (timesteps) and sort
                timesteps = sorted(other_data.keys(), key=int)

                # Get number of altitude shells (assuming all timesteps have the same length)
                num_altitude_shells = len(other_data[timesteps[0]]["ror"])

                # Prepare the figure with 4 subplots now
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))

                # Color map for time evolution
                colors = cm.viridis(np.linspace(0, 1, len(timesteps)))

                # Static Plot: loop through each timestep and plot each metric
                for idx, timestep in enumerate(timesteps):
                        ror = other_data[timestep]["ror"]
                        collision_prob = other_data[timestep]["collision_probability"]
                        launch_rate = other_data[timestep]["launch_rate"]
                        excess_returns = other_data[timestep]["excess_returns"]

                        axes[0].plot(range(num_altitude_shells), ror, color=colors[idx], label=f"Year {timestep}")
                        axes[1].plot(range(num_altitude_shells), collision_prob, color=colors[idx])
                        axes[2].plot(range(num_altitude_shells), launch_rate, color=colors[idx])
                        axes[3].plot(range(num_altitude_shells), excess_returns, color=colors[idx])

                axes[0].set_title("Rate of Return (RoR)")
                axes[1].set_title("Collision Probability")
                axes[2].set_title("Launch Rate")
                axes[3].set_title("Excess Returns")

                # Set labels and ticks for each subplot
                for ax in axes:
                        ax.set_xlabel("Shell - Mid Altitude (km)")
                        ax.set_ylabel("Value")
                        ax.set_xticklabels(self.HMid)

                # Add a legend to the figure (outside the individual axes)
                fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)

                plt.tight_layout()

                # Save the combined plot
                combined_file_path = os.path.join(plot_data.path, "combined_time_evolution.png")
                plt.savefig(combined_file_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"All economic metrics single plot saved to {combined_file_path}")


        def ror_cp_and_launch_rate_gif(self, plot_data, other_data):
                """
                Generate and save an animated plot for the time evolution of different parameters 
                (RoR, Collision Probability, Launch Rate, Excess Returns).
                """
                # Extract keys (timesteps) and sort
                timesteps = sorted(other_data.keys(), key=int)

                # Get number of altitude shells (assuming all timesteps have the same length)
                num_altitude_shells = len(other_data[timesteps[0]]["ror"])

                # Determine global min/max for each metric across all timesteps
                ror_values = [val for timestep in timesteps for val in other_data[timestep]["ror"]]
                collision_values = [val for timestep in timesteps for val in other_data[timestep]["collision_probability"]]
                launch_values = [val for timestep in timesteps for val in other_data[timestep]["launch_rate"]]
                excess_values = [val for timestep in timesteps for val in other_data[timestep]["excess_returns"]]

                ror_min, ror_max = min(ror_values), max(ror_values)
                collision_min, collision_max = min(collision_values), max(collision_values)
                launch_min, launch_max = min(launch_values), max(launch_values)
                excess_min, excess_max = min(excess_values), max(excess_values)

                # Create the figure and axes with 4 subplots
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))

                def update(frame):
                        timestep = timesteps[frame]
                        ror = other_data[timestep]["ror"]
                        collision_prob = other_data[timestep]["collision_probability"]
                        launch_rate = other_data[timestep]["launch_rate"]
                        excess_returns = other_data[timestep]["excess_returns"]

                        # Clear each axis before plotting new data
                        for ax in axes:
                                ax.clear()

                        # Plot each metric with fixed y-axis limits
                        axes[0].plot(range(num_altitude_shells), ror, color='b')
                        axes[0].set_ylim(ror_min, ror_max)
                        axes[0].set_title(f"Rate of Return (RoR) - Year {timestep}")
                        axes[0].set_xlabel("Shell - Mid Altitude (km)")
                        axes[0].set_ylabel("RoR")
                        axes[0].set_xticks(range(len(self.HMid)))
                        axes[0].set_xticklabels(self.HMid)

                        axes[1].plot(range(num_altitude_shells), collision_prob, color='r')
                        axes[1].set_ylim(collision_min, collision_max)
                        axes[1].set_title(f"Collision Probability - Year {timestep}")
                        axes[1].set_xlabel("Shell - Mid Altitude (km)")
                        axes[1].set_ylabel("Collision Probability")
                        axes[1].set_xticks(range(len(self.HMid)))
                        axes[1].set_xticklabels(self.HMid)

                        axes[2].plot(range(num_altitude_shells), launch_rate, color='g')
                        axes[2].set_ylim(launch_min, launch_max)
                        axes[2].set_title(f"Launch Rate - Year {timestep}")
                        axes[2].set_xlabel("Shell - Mid Altitude (km)")
                        axes[2].set_ylabel("Launch Rate")
                        axes[2].set_xticks(range(len(self.HMid)))
                        axes[2].set_xticklabels(self.HMid)

                        axes[3].plot(range(num_altitude_shells), excess_returns, color='m')
                        axes[3].set_ylim(excess_min, excess_max)
                        axes[3].set_title(f"Excess Returns - Year {timestep}")
                        axes[3].set_xlabel("Shell - Mid Altitude (km)")
                        axes[3].set_ylabel("Excess Returns")
                        axes[3].set_xticks(range(len(self.HMid)))
                        axes[3].set_xticklabels(self.HMid)

                        plt.tight_layout()

                # Create the animation
                ani = animation.FuncAnimation(fig, update, frames=len(timesteps), repeat=True)

                # Save as GIF
                combined_file_path = os.path.join(plot_data.path, "space_metrics_evolution.gif")
                ani.save(combined_file_path, writer="pillow", fps=2)

                plt.close()
                print(f"Animated evolution plot saved to {combined_file_path}")


        # def collision_probability_stacked_by_species(self, plot_data, other_data):
        #         pass

        # def pmd_effectiveness(self, plot_data, other_data):
        #         pass