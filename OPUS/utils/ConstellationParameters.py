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

    def define_initial_launch_rate(self, MOCAT):
        """
            Defines the initial launch rate for a given constellation.
            This takes the x0 of the model and the mocat_species defined for the constellation by the user. 

            Args:
                MOCAT (Model): The MOCAT model
        """

        x0 = MOCAT.scenario_properties.x0.T.values.flatten() # x0 starts off in a df format. 

        # The old version assumed that there was always one species of satellite in the model. 
        # Now the species name must be used to find the initial population of slotted objects. This will need to change

        
        print("Cost function parameters calculated")

        Si = x0[0:MOCAT.scenario_properties.n_shells] # Initial population of slotted objects

        lam = np.zeros((len(Si), 1))
        lam[:, 0] = 1 / 5 * Si # 5 should be replaced with mocat.scenario_properties.dt which is the operational lifetime of a satellite

        # Modify launch rate based on constellation parameters
        for i in range(self.n_constellations):
            location_index = self.location_index[i]
            final_size = self.final_size[i]
            linear_rate = self.linear_rate[i]
            
            lam[location_index, 0] = self.constellation_buildup(location_index, final_size, linear_rate, Si)
        
        self.lam = lam
        return self.lam

        

    def constellation_buildup(self, location_index, final_size, linear_rate, Si):
        """
            Sets the launch rate for a gien constellation at a given location

            Args:
                location_index (int): The location index of the constellation
                final_size (int): The final size of the constellation
                linear_rate (float): The linear rate of the constellation
                Si (numpy.ndarray): Initial population of slotted objects
        """

        current_size = Si[location_index]

        remaining_size = max(final_size - current_size, 0)

        return min(remaining_size, linear_rate)

