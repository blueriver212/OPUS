from pyssem.model import Model

class MultiSpecies:
    """
    This will host each species that are required for the global optimisation. 
    It will store all things for OPUS. 
    """
    def __init__(self, species_names):
        """
        Constructor for the MultiSpecies class.

        Parameters
        ----------
        species : list
            List of species objects.
        scenario_properties : ScenarioProperties
            Scenario properties object.
        """
        # Create a list of species objects with the name of the species
        self.species = [OPUSSpecies(name) for name in species_names]

    def get_species_position_indexes(self, MOCAT: Model):
        """
        Assigns start and end slice indices to each species and their PMD-linked species 
        in a MultiSpecies object.

        Parameters
        ----------
        MOCAT : MOCATModel
            The MOCAT model containing scenario properties like species names and number of shells.
        multispecies : MultiSpecies
            An object containing a list of OPUSSpecies objects.
        """
        n_shells = MOCAT.scenario_properties.n_shells
        species_names = MOCAT.scenario_properties.species_names

        for obj in self.species:
            # Main species slices
            if obj.name not in species_names:
                raise ValueError(f"Species name '{obj.name}' not found in MOCAT scenario properties.")
            
            obj.species_idx = species_names.index(obj.name)
            obj.start_slice = obj.species_idx * n_shells
            obj.end_slice = obj.start_slice + n_shells

            # PMD-linked species slices (optional)
            if hasattr(obj, "pmd_linked_species") and obj.pmd_linked_species:
                if obj.pmd_linked_species not in species_names:
                    raise ValueError(f"PMD-linked species '{obj.pmd_linked_species}' not found in MOCAT scenario properties.")
                
                derelict_idx = species_names.index(obj.pmd_linked_species)
                obj.derelict_start_slice = derelict_idx * n_shells
                obj.derelict_end_slice = obj.derelict_start_slice + n_shells

    def get_mocat_species_parameters(self, MOCAT: Model):
        """
            This function will find the MOCAT species for the fringe satellite and then 
            abstract any information required for OPUS modelling. 
            
            Specifcially useful for the Solver, e.g DeltaT for Lifetime Assessment. 
        """
        values_to_extract = ['deltat', 'mass', 'Pm']
        for opus_species in self.species:
            for species_group in MOCAT.scenario_properties.species.values():
                for mocat_species in species_group:
                    if opus_species.name == mocat_species.sym_name:
                        for attr in values_to_extract:
                            try:
                                value = getattr(mocat_species, attr)
                                setattr(opus_species, attr, value)
                            except AttributeError:
                                print(f"Warning: '{attr}' not found in MOCAT species '{mocat_species.sym_name}'")


    def increase_demand(self):
        """
        This method will look through the species, if there is a demand growth value for any of the species. The revenue will be increased proportionally. 
        """
        for species in self.species:
            if species.econ_params.demand_growth is not None:
                species.econ_params.intercept = species.econ_params.intercept * (1 + species.econ_params.demand_growth)
                print("intercept now: ", species.econ_params.intercept)

        
class OPUSSpecies:
    """
    This class is used to create a species object. 
    It will be used to create the species in the MOCAT model. 
    """

    def __init__(self, name):
        """
        Constructor for the OPUSSpecies class.

        Parameters
        ----------
        name : str
            Name of the species.
        """
        self.name = name
        self.econ_params = None