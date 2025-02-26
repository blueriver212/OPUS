import pandas as pd
import numpy as np

class ConstellationParameters:

    def __init__(self, filename):
        
        # read the constellation csv
        df = pd.read_csv(filename)

        # Initialise the parameters
        self.n_constellations = int(df['n_constellations'][0])
        self.final_size = df['target_sizes'].tolist()
        self.linear_rate = df['max_launch_rates'].tolist()
        self.mocat_species = df['mocat_species'].to_string()
        self.altitude = df['altitude'].tolist()

    def define_initial_launch_rate(self, MOCAT, constellation_start_slice, constellation_end_slice, x0):
        """
            Defines the initial launch rate for a given constellation.
            This takes the x0 of the model and the mocat_species defined for the constellation by the user. 

            Args:
                MOCAT (Model): The MOCAT model
                sats_idx (int): The index of the species
        """

        # The old version assumed that there was always one species of satellite in the model. 
        # Now the species name must be used to find the initial population of slotted objects. This will need to change
        
        print("Cost function parameters calculated")

        Si = x0[constellation_start_slice:constellation_end_slice] # Initial population of slotted objects

        # Initialize lam with None values
        lam = [None] * len(x0)

        # Modify launch rate based on constellation parameters
        for i in range(self.n_constellations):
            final_size = self.final_size[i]
            linear_rate = self.linear_rate[i]
            altitude = self.altitude[i]

            # Calculate the location index based on the altitude using MOCAT.scenario_properties.R02
            location_index = np.argmin(np.abs(MOCAT.scenario_properties.R0_km - altitude))
            
            # Assign launch rate only to the specified species
            if constellation_start_slice <= location_index < constellation_end_slice:
                lam[location_index] = self.constellation_buildup(location_index, final_size, linear_rate, Si)
        
        # Assign initial launch rate for the specified species
        lam[constellation_start_slice:constellation_end_slice] = (1 / 5) * Si  # Should be MOCAT.scenario_properties.dt
     
        self.lam = lam
        return self.lam

    def constellation_buildup(self, location_index, final_size, linear_rate, Si):
        """
            Sets the launch rate for a given constellation at a given location

            Args:
                location_index (int): The location index of the constellation
                final_size (int): The final size of the constellation
                linear_rate (float): The linear rate of the constellation
                Si (numpy.ndarray): Initial population of slotted objects
        """

        current_size = Si[location_index]

        remaining_size = max(final_size - current_size, 0)

        return min(remaining_size, linear_rate)
    
    def constellation_launch_rate_for_next_period(self, MOCAT, lam, species_start_index, species_end_index, Si):
        """
            Sets the launch rate for a given constellation at a given location for the next period

            Args:
                location_index (int): The location index of the constellation
                final_size (int): The final size of the constellation
                linear_rate (float): The linear rate of the constellation
                Si (numpy.ndarray): Initial population of slotted objects
        """

        for i in range(self.n_constellations):
            final_size = self.final_size[i]
            linear_rate = self.linear_rate[i]
            altitude = self.altitude[i]

            # Calculate the location index based on the altitude using MOCAT.scenario_properties.R02
            location_index = np.argmin(np.abs(MOCAT.scenario_properties.R0_km - altitude))
            
            # Assign launch rate only to the specified species
            if species_start_index <= location_index < species_end_index:
                lam[location_index] = [self.constellation_buildup(location_index, final_size, linear_rate, Si)]
        
        # Assign initial launch rate for the specified species
        for idx in range(species_start_index, species_end_index):
            lam[idx] = [(1 / 5 * Si[idx])] # should be MOCAT.scenario_properties.dt
