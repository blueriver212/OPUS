import os
import json
import numpy as np
import matplotlib.pyplot as plt

class PlotData:
        """
                This will hold in the data required to make any of the plots. 
                It will also be passed to each of the plotting functions.       
        """

        def __init__(self, scenario, path, MOCAT):
                self.scenario = scenario
                self.path = path

                # Load the data from the scenario folder
                self.data = self.load_data(path)

                self.species_names = MOCAT.scenario_properties.species_names
                self.n_shells = MOCAT.scenario_properties.n_shells
                self.n_species = MOCAT.scenario_properties.species_length
                self.HMid = MOCAT.scenario_properties.HMid

        def load_data(self, path):
                """
                        Load the data from the scenario folder
                """

                if not os.path.exists(path):
                        print(f"Error: {path} does not exist.")
                        return None
                
                # find the json file in the folder and load this
                json_file = [f for f in os.listdir(path) if f.endswith(".json")]

                if len(json_file) == 0:
                        print(f"Error: No json file found in {path}.")
                        return None
                elif len(json_file) > 1:
                        print(f"Error: More than one json file found in {path}. You need to specify which file should be used for plotting.")
                        return None
                
                with open(os.path.join(path, json_file[0]), "r") as f:
                        data = json.load(f)
                        return data 
 
class PlotHandler:  
        def __init__(self, MOCAT, scenario_files, simulation_name, plot_types=["all_plots"]):
                
                self.MOCAT = MOCAT
                self.scenario_files = scenario_files # This will be a list of each sub-scenario run name
                self.simulation_name = simulation_name # This is the overall name of the simualtion 
                self.plot_types = plot_types # This will be a list of the types of plots to be generated

                # This will rely on the fact that there is a file available under the simulation name in the Results folder. 
                self.simulation_folder = os.path.join("Results", self.simulation_name)
                
                # if not show error to the user
                if not os.path.exists(self.simulation_folder):
                        print(f"Error: {self.simulation_folder} does not exist.")
                        return
                
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


                                # If the plot_types is None, then generate all plots
                                if "all_plots" in self.plot_types:
                                        self.all_plots(plot_data)
                                else:
                                        # Dynamically generate plots
                                        for plot_name in self.plots:
                                                plot_method = getattr(self, plot_name, None)
                                                if callable(plot_method):
                                                        print(f"Creating plot: {plot_name}")
                                                        plot_method()
                                                else:
                                                        print(f"Warning: Plot '{plot_name}' not found. Skipping...")

        def all_plots(self, plot_data):
                """
                Run all plot functions, irrespective of the plots list.
                """
                for attr_name in dir(self):
                        if callable(getattr(self, attr_name)) and attr_name not in ("__init__", "all_plots"):
                                if not attr_name.startswith("_"):
                                        print(f"Creating plot: {attr_name}")
                                        plot_method = getattr(self, attr_name)
                                        plot_method(plot_data)


        def count_by_shell_and_time_per_species(self, plot_data):
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
                        plt.yticks(ticks=range(data.shape[1]), labels=plot_data.HMid)

                        # Save the plot to the designated folder
                        file_path = os.path.join(plot_data.path, f"count_over_time_{sp}.png")
                        plt.savefig(file_path, dpi=300, bbox_inches='tight')
                        plt.close()

        def combined_count_by_shell_and_time(self, plot_data):
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
                        ax.set_yticklabels(plot_data.HMid)
                        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

                # Turn off unused subplots
                for ax in axes[len(species_data):]:
                        ax.axis('off')

                # Save the combined plot
                combined_file_path = os.path.join(plot_data.path, "combined_species_heatmaps.png")
                plt.savefig(combined_file_path, dpi=300, bbox_inches='tight')
                plt.close()

        def collision_probability_stacked_by_species(self, plot_data):
                pass

        def cost_function_vs_time(self, plot_data):
                pass

        def revenue_vs_time(self, plot_data):
                pass

        def pmd_effectiveness(self, plot_data):
                pass

        def comparison_plots(self, plot_data):
                pass
                


