import os
import json
import numpy as np

class PostProcessing:
    def __init__(self, MOCAT, scenario_name, simulation_name, species_data, other_results, econ_params, grid_search=False):
        self.MOCAT = MOCAT
        self.scenario_name = scenario_name # this is the breadkdown of the scenario
        self.simulation_name = simulation_name # this is the overall name of the simulation
        self.species_data = species_data
        self.econ_params = econ_params

        # Other results will have this form
        # simulation_results[time_idx] = {
        #         "ror": ror,
        #         "collision_probability": collision_probability,
        #         "launch_rate" : launch_rate
        #     }
        self.other_results = other_results
        
        if not grid_search:
            self.create_folder_structure()
            self.post_process_data()
            self.post_process_economic_data(self.econ_params)
        else:
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

        serializable_species_data = {sp: {year: data.tolist() for year, data in self.species_data[sp].items()} for sp in self.species_data.keys()}
        # Save the serialized data to a JSON file in the appropriate folder
        output_path = f"./Results/{self.simulation_name}/{self.scenario_name}/species_data_{self.scenario_name}.json"
        with open(output_path, 'w') as json_file:
            json.dump(serializable_species_data, json_file, indent=4)

        print(f"species_data has been exported to {output_path}")

        # serializable_other_results = {
        #     int(time_idx): {
        #         "ror": data["ror"].tolist() if isinstance(data["ror"], (list, np.ndarray)) else data["ror"],
        #         "collision_probability": data["collision_probability"].tolist() if isinstance(data["collision_probability"], (list, np.ndarray)) else data["collision_probability"],
        #         "launch_rate": data["launch_rate"].tolist() if isinstance(data["launch_rate"], (list, np.ndarray)) else data["launch_rate"],
        #         "collision_probability_all_species": data["collision_probability_all_species"].tolist() if isinstance(data["collision_probability_all_species"], (list, np.ndarray)) else data["collision_probability_all_species"],
        #         "umpy": data["umpy"], 
        #         "excess_returns": data["excess_returns"].tolist() if isinstance(data["excess_returns"], (list, np.ndarray)) else data["excess_returns"],
        #         "non_compliance": {
        #             sp: val.tolist() if isinstance(val, (list, np.ndarray)) else val
        #             for sp, val in data["non_compliance"].items()
        #         } if isinstance(data["non_compliance"], dict) else data["non_compliance"]
        #     }
        #     for time_idx, data in self.other_results.items()
        # }
        def convert_to_serializable(obj):
            """Recursively convert numpy arrays and other non-serializable objects to JSON-serializable format."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_other_results = {
            int(time_idx): convert_to_serializable(data)
            for time_idx, data in self.other_results.items()
        }

        other_results_output_path = f"./Results/{self.simulation_name}/{self.scenario_name}/other_results_{self.scenario_name}.json"
        
        with open(other_results_output_path, 'w') as json_file:
            json.dump(serializable_other_results, json_file, indent=4)

        print(f"other_results has been exported to {other_results_output_path}")

    def post_process_economic_data(self, econ_params):
        """
        Writes all key economic parameters from the econ_params object to a JSON file,
        automatically extracting attributes from the class.
        """
        # Build the output path
        other_results_output_path = (
            f"./Results/{self.simulation_name}/{self.scenario_name}/"
            f"econ_params_{self.scenario_name}.json"
        )
        os.makedirs(os.path.dirname(other_results_output_path), exist_ok=True)
        
        def convert_value(val):
            """Convert NumPy arrays and scalars to native Python types."""
            if isinstance(val, np.ndarray):
                return val.tolist()
            elif isinstance(val, (np.integer, np.int32, np.int64)):
                return int(val)
            elif isinstance(val, (np.floating, np.float32, np.float64)):
                return float(val)
            else:
                return val

        # Create a dictionary by iterating over econ_params attributes
        data_to_save = {}
        for key, value in econ_params.__dict__.items():
            # Skip private attributes and callables
            if key.startswith('_') or callable(value):
                continue
            # Optionally, skip attributes that are not part of the economic parameters
            if key == "mocat":
                continue

            data_to_save[key] = convert_value(value)

        # Write the dictionary to a JSON file
        with open(other_results_output_path, "w") as outfile:
            json.dump(data_to_save, outfile, indent=4)

        print(f"Economic parameters written to {other_results_output_path}")



