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
        self.econ_params = None

    def get_species_position_indexes(self, MOCAT):
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
            
            species_idx = species_names.index(obj.name)
            obj.start_slice = species_idx * n_shells
            obj.end_slice = obj.start_slice + n_shells

            # PMD-linked species slices (optional)
            if hasattr(obj, "pmd_linked_species") and obj.pmd_linked_species:
                if obj.pmd_linked_species not in species_names:
                    raise ValueError(f"PMD-linked species '{obj.pmd_linked_species}' not found in MOCAT scenario properties.")
                
                derelict_idx = species_names.index(obj.pmd_linked_species)
                obj.derelict_start_slice = derelict_idx * n_shells
                obj.derelict_end_slice = obj.derelict_start_slice + n_shells


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