import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

class PlotData:
        """
                This will hold in the data required to make any of the plots. 
                It will also be passed to each of the plotting functions.       
        """

        def __init__(self, scenario, path, MOCAT):
                self.scenario = scenario
                self.path = path

                # Load the data from the scenario folder
                self.data, self.other_data = self.load_data(path)

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
                elif len(json_files) > 2:
                        print(f"Error: More than two JSON files found in {path}. Only two files are expected.")
                        return None
                
                # Initialize variables for storing the two data files
                data = None
                other_data = None
                
                # Loop through the files and assign them based on content
                for file in json_files:
                        file_path = os.path.join(path, file)
                        with open(file_path, "r") as f:
                                file_content = json.load(f)
                                
                                if "species_data" in file_path:
                                        data = file_content  # First, assign the species data
                                elif "other_results" in file_path:
                                        other_data = file_content  # Then, assign the other results data
                
                # Check if both data files were found
                if data is None:
                        print(f"Error: No file containing 'species_data' found in {path}.")
                if other_data is None:
                        print(f"Error: No file containing 'other_results' found in {path}.")
                
                return data, other_data


class PlotHandler:  
        def __init__(self, MOCAT, scenario_files, simulation_name, plot_types=["all_plots"]):
                
                self.MOCAT = MOCAT
                self.scenario_files = scenario_files # This will be a list of each sub-scenario run name
                self.simulation_name = simulation_name # This is the overall name of the simualtion 
                self.plot_types = plot_types # This will be a list of the types of plots to be generated
                self.HMid = self.MOCAT.scenario_properties.HMid

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
                                other_data = plot_data.get_other_data()

                                # If the plot_types is None, then generate all plots
                                if "all_plots" in self.plot_types:
                                        self.all_plots(plot_data, other_data)
                                else:
                                        # Dynamically generate plots
                                        for plot_name in self.plots:
                                                plot_method = getattr(self, plot_name, None)
                                                if callable(plot_method):
                                                        print(f"Creating plot: {plot_name}")
                                                        plot_method()
                                                else:
                                                        print(f"Warning: Plot '{plot_name}' not found. Skipping...")

        def all_plots(self, plot_data, other_data):
                """
                Run all plot functions, irrespective of the plots list.
                """
                for attr_name in dir(self):
                        if callable(getattr(self, attr_name)) and attr_name not in ("__init__", "all_plots"):
                                if not attr_name.startswith("_"):
                                        print(f"Creating plot: {attr_name}")
                                        plot_method = getattr(self, attr_name)
                                        plot_method(plot_data, other_data)


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


        def collision_probability_stacked_by_species(self, plot_data, other_data):
                pass

        def cost_function_vs_time(self, plot_data, other_data):
                pass

        def revenue_vs_time(self, plot_data, other_data):
                pass

        def pmd_effectiveness(self, plot_data, other_data):
                pass

        def comparison_plots(self, plot_data, other_data):
                pass
                


