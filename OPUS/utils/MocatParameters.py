from pyssem.model import Model # mocat-ssem
from .MultiSpecies import MultiSpecies
from .EconParameters import EconParameters
import json 
import numpy as np

def configure_mocat(MOCAT_config: json, multi_species: MultiSpecies = None) -> Model:
    """
        Configure's MOCAT-pySSEM model with a provided input json. 
        To find a correct configuration, please refer to the MOCAT documentation. https://github.com/ARCLab-MIT/pyssem/

        Args:
            MOCAT_config (json): Dictionary containing the MOCAT configuration parameters.

        Returns:
            Model: An configured instance of the MOCAT model.
    """
    scenario_props = MOCAT_config["scenario_properties"]
    # Create an instance of the pySSEM_model with the simulation parameters
    MOCAT = Model(
        start_date=scenario_props["start_date"].split("T")[0],  # Assuming the date is in ISO format
        simulation_duration=scenario_props["simulation_duration"],
        steps=scenario_props["steps"],
        min_altitude=scenario_props["min_altitude"],
        max_altitude=scenario_props["max_altitude"],
        n_shells=scenario_props["n_shells"],
        launch_function=scenario_props["launch_function"],
        integrator=scenario_props["integrator"],
        density_model=scenario_props["density_model"],
        LC=scenario_props["LC"],
        v_imp = scenario_props.get("v_imp", None), 
        fragment_spreading=scenario_props.get("fragment_spreading", False),
        parallel_processing=scenario_props.get("parallel_processing", True),
        baseline=scenario_props.get("baseline", False),
        indicator_variables=scenario_props.get("indicator_variables", None),
        launch_scenario=scenario_props["launch_scenario"],
        SEP_mapping=MOCAT_config["SEP_mapping"] if "SEP_mapping" in MOCAT_config else None,
        elliptical=scenario_props.get("elliptical", False),
        eccentricity_bins=scenario_props.get("eccentricity_bins", None)
    )

    species = MOCAT_config["species"]

    MOCAT.configure_species(species)
    # Create an active_loss_setup for each of the species in the model.
    if multi_species != None:
        for species in multi_species.species:
            MOCAT.opus_active_loss_setup(species.name)

    if MOCAT.scenario_properties.elliptical:
        MOCAT.build_model(elliptical=True)
    else:
        MOCAT.build_model(elliptical=False)

    print("You have these species in the model: ", MOCAT.scenario_properties.species_names)

    # Find the PMD linked species and return the index. 
    if multi_species == None:
        return MOCAT

    for opus_species in multi_species.species:
        pmd_linked_species_to_fringe = [
            species
            for species_group in MOCAT.scenario_properties.species.values()
            for species in species_group
            if any(linked_species.sym_name == opus_species.name for linked_species in species.pmd_linked_species)
        ]

        if len(pmd_linked_species_to_fringe) != 1:
            raise ValueError("Please ensure that there is only one species linked to the fringe satellite.")
        else:
            opus_species.pmd_linked_species = pmd_linked_species_to_fringe[0].sym_name

    # Match the econ parameters from the json to the multispecies object.
    for dict in MOCAT_config["species"]:
        try:
            opus_params = dict['OPUS']
            sym_name = dict['sym_name']
            for species in multi_species.species:
                if species.name == sym_name:
                    species.econ_params = EconParameters(opus_params, MOCAT)
        except KeyError:
            # If the sym_name is in the multi_species object, it means it has been asked to be econ parameterized. Used the default values.
            for obj in multi_species.species:
                if obj.name == dict['sym_name']:
                    obj.econ_params = EconParameters({}, MOCAT)
                    print("The species: ", dict['sym_name'], " is not econ parameterized. Using the default values.")
                    break
            print("Using the default economic parameters for the species: ", dict['sym_name'])
            print(f"Key 'OPUS' not found in the dictionary for species '{dict['sym_name']} \n Please include if you want to use the economic parameters in the model.")

 
    return MOCAT, multi_species