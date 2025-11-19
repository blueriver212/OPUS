import os
import json
import numpy as np

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

