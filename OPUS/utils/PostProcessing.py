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
        self.other_results = other_results
        
        # --- DEFINE ALL PATHS HERE ---
        self.base_output_path = f"./Results/{self.simulation_name}/{self.scenario_name}"
        self.species_data_path = os.path.join(self.base_output_path, f"species_data_{self.scenario_name}.json")
        self.other_results_path = os.path.join(self.base_output_path, f"other_results_{self.scenario_name}.json")
        self.econ_params_path = os.path.join(self.base_output_path, f"econ_params_{self.scenario_name}.json")
        # --- END OF PATH DEFINITIONS ---
        
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
        if not os.path.exists(self.base_output_path):
            os.makedirs(self.base_output_path)

    def post_process_data(self):
        """
            Create plots for the simulation. If all plots, create all.
        """

        serializable_species_data = {sp: {year: data.tolist() for year, data in self.species_data[sp].items()} for sp in self.species_data.keys()}
        # Save the serialized data to a JSON file in the appropriate folder
        with open(self.species_data_path, 'w') as json_file:
            json.dump(serializable_species_data, json_file, indent=4)

        print(f"species_data has been exported to {self.species_data_path}")

        def convert_to_serializable(obj):
            """Recursively convert numpy arrays and other non-serializable objects to JSON-serializable format."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # Add handlers for numpy numeric types
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
        
        with open(self.other_results_path, 'w') as json_file:
            json.dump(serializable_other_results, json_file, indent=4)

        print(f"other_results has been exported to {self.other_results_path}")

    def post_process_economic_data(self, all_econ_params):
        """
        Processes the economic parameters from all species and saves them to a JSON file.
        
        Args:
            all_econ_params (dict): A dictionary where keys are species names 
                                  and values are their EconParameters objects.
        """
        # Main dictionary we save to JSON
        all_species_econ_data = {}

        try:
            # Loop through the dictionary received
            for species_name, econ_object in all_econ_params.items():
                
                # This will hold the data for one species
                single_species_data = {}
                
                for key, value in econ_object.__dict__.items():
                    
                    if key == 'mocat':
                        # Don't try to serialize the entire MOCAT model
                        continue
                        
                    if isinstance(value, np.ndarray):
                        single_species_data[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.int64)):
                        single_species_data[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64)):
                        single_species_data[key] = float(value)
                    
                    elif isinstance(value, (int, float, str, list, dict, type(None))):
                        single_species_data[key] = value
                    else:
                        # As a fallback, convert any other types to string
                        single_species_data[key] = str(value)

                # Add this species' data to the main dictionary
                all_species_econ_data[species_name] = single_species_data
            
            # Save the new nested dictionary
            with open(self.econ_params_path, "w") as f:
                json.dump(all_species_econ_data, f, indent=4)
            print(f"econ_params has been exported to {self.econ_params_path}")

        except Exception as e:
            print(f"Error exporting economic data in post_process_economic_data: {e}")
            if isinstance(e, AttributeError) and 'NoneType' in str(e):
                print("...This may be because an active species in multi_species_names does not have an 'OPUS' block in the JSON config.")