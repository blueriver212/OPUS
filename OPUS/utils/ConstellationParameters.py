import pandas as pd
import numpy as np

class ConstellationParameters:

    def __init__(self, filename):
        
        # read the constellation csv
        df = pd.read_csv(filename)

        # Initialise the parameters
        self.n_constellations = int(df['n_constellations'][0])
        self.location_index = df['location_indices'].tolist()
        self.final_size = df['target_sizes'].tolist()
        self.linear_rate = df['max_launch_rates'].tolist()
        self.mocat_species = df['mocat_species'].to_string()
        self.altitude = df['altitude'].tolist()

    def define_initial_launch_rate(self, MOCAT, sats_idx):
        """
            Defines the initial launch rate for a given constellation.
            This takes the x0 of the model and the mocat_species defined for the constellation by the user. 

            Args:
                MOCAT (Model): The MOCAT model
                sats_idx (int): The index of the species
        """

        x0 = MOCAT.scenario_properties.x0.T.values.flatten() # x0 starts off in a df format. 
        # The old version assumed that there was always one species of satellite in the model. 
        # Now the species name must be used to find the initial population of slotted objects. This will need to change
        
        print("Cost function parameters calculated")

        n_shells = MOCAT.scenario_properties.n_shells
        Si = x0[0:n_shells] # Initial population of slotted objects

        # Initialize lam with None values
        lam = [None] * len(x0)

        # Define the start and end indices for the species using sats_idx
        if sats_idx == 0:
            species_start_index = 0
            species_end_index = n_shells
        else:
            species_start_index = (sats_idx - 1) * n_shells
            species_end_index = sats_idx * n_shells

        # Modify launch rate based on constellation parameters
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
        

    def fringe_sat_pop_feedback_controller():
        pass


    def open_acces_solver():
        pass

