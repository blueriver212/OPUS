import os
import json
import numpy as np

class PostProcessing:
    def __init__(self, MOCAT, scenario_name, simulation_name, species_data, other_results, econ_params):
        self.MOCAT = MOCAT
        self.scenario_name = scenario_name # this is the breadkdown of the scenario
        self.simulation_name = simulation_name # this is the overall name of the simulation
        self.species_data = species_data
        self.econ_params = econ_params
        self.umpy_score = None # sammie addition
        self.adr_dict = {}

        # Other results will have this form
        # simulation_results[time_idx] = {
        #         "ror": ror,
        #         "collision_probability": collision_probability,
        #         "launch_rate" : launch_rate
        #     }
        self.other_results = other_results

        self.create_folder_structure()
        self.post_process_data()
        self.post_process_economic_data(self.econ_params)

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
                "launch_rate": data["launch_rate"].tolist() if isinstance(data["launch_rate"], (list, np.ndarray)) else data["launch_rate"],
                "collision_probability_all_species": data["collision_probability_all_species"].tolist() if isinstance(data["collision_probability_all_species"], (list, np.ndarray)) else data["collision_probability_all_species"],
                "umpy": data["umpy"], 
                "excess_returns": data["excess_returns"].tolist() if isinstance(data["excess_returns"], (list, np.ndarray)) else data["excess_returns"],
                "ICs": data["ICs"].tolist() if isinstance(data["ICs"], (list, np.ndarray)) else data["ICs"], #sammie addition
                "tax_revenue_total": data["tax_revenue_total"],
                "tax_revenue_by_shell": data["tax_revenue_by_shell"].tolist() if isinstance(data["tax_revenue_by_shell"], np.ndarray) else data["tax_revenue_by_shell"],
                "welfare": data.get("welfare",0),
            }
            for time_idx, data in self.other_results.items()
        }

        other_results_output_path = f"./Results/{self.simulation_name}/{self.scenario_name}/other_results_{self.scenario_name}.json"
        
        with open(other_results_output_path, 'w') as json_file:
            json.dump(serializable_other_results, json_file, indent=4)

        final_key = list(self.other_results.keys())[-1]
        # final_results = self.other_results.keys()[-1]

        test = "test"
        final_umpy = np.sum(self.other_results[final_key]["umpy"])
        score = final_umpy #* (-1)
        umpy_path = f"./Results/{self.simulation_name}/{self.scenario_name}/final_umpy.json"
        if not os.path.exists(os.path.dirname(umpy_path)):
            os.makedirs(os.path.dirname(umpy_path))
        with open(umpy_path, 'w') as json_file:
            json.dump(score, json_file, indent=4)
        self._umpy_score = score

        self._adr_dict[self.scenario_name] = self._umpy_score
        print(f"other_results has been exported to {other_results_output_path}")

    @property
    def umpy_score(self):
        return self._umpy_score
    
    @umpy_score.setter
    def umpy_score(self, value, deep=True):
        self._umpy_score = value

    @property
    def adr_dict(self):
        return self._adr_dict
    
    @adr_dict.setter
    def adr_dict(self, value):
        self._adr_dict = value

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



