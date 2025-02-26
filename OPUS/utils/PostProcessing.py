import os
import json
import numpy as np

class PostProcessing:
    def __init__(self, MOCAT, scenario_name, simulation_name, species_data, other_results):
        self.MOCAT = MOCAT
        self.scenario_name = scenario_name # this is the breadkdown of the scenario
        self.simulation_name = simulation_name # this is the overall name of the simulation
        self.species_data = species_data

        # Other results will have this form
        # simulation_results[time_idx] = {
        #         "ror": ror,
        #         "collision_probability": collision_probability,
        #         "launch_rate" : launch_rate
        #     }
        self.other_results = other_results

        self.create_folder_structure()
        self.post_process_data()

    def create_folder_structure(self):
        """
            Create the folder structure for the simulation
        """
        # Create the folder structure
        if not os.path.exists(f"./Results/{self.simulation_name}"):
            os.makedirs(f"./Results/{self.simulation_name}")
        if not os.path.exists(f"./Results/{self.simulation_name}/{self.scenario_name}"):
            os.makedirs(f"./Results/{self.simulation_name}/{self.scenario_name}")

    def post_process_data(self):
        """
            Create plots for the simulation. If all plots, create all.
        """

        serializable_species_data = {sp: data.tolist() for sp, data in self.species_data.items()}

        # Save the serialized data to a JSON file in the appropriate folder
        output_path = f"./Results/{self.simulation_name}/{self.scenario_name}/species_data_{self.scenario_name}.json"
        with open(output_path, 'w') as json_file:
            json.dump(serializable_species_data, json_file, indent=4)

        print(f"species_data has been exported to {output_path}")

        serializable_other_results = {
            int(time_idx): {
                "ror": data["ror"].tolist() if isinstance(data["ror"], (list, np.ndarray)) else data["ror"],
                "collision_probability": data["collision_probability"].tolist() if isinstance(data["collision_probability"], (list, np.ndarray)) else data["collision_probability"],
                "launch_rate": data["launch_rate"].tolist() if isinstance(data["launch_rate"], (list, np.ndarray)) else data["launch_rate"]
            }
            for time_idx, data in self.other_results.items()
        }

        other_results_output_path = f"./Results/{self.simulation_name}/{self.scenario_name}/other_results_{self.scenario_name}.json"
        
        with open(other_results_output_path, 'w') as json_file:
            json.dump(serializable_other_results, json_file, indent=4)

        print(f"other_results has been exported to {other_results_output_path}")



