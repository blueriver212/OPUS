from pyssem.model import Model # mocat-ssem
import json 
import numpy as np

def configure_mocat(MOCAT_config: json, fringe_satellite: str = None) -> Model:
    """
        Configure's MOCAT-pySSEM model with a provided input json. 
        To find a correct configuration, please refer to the MOCAT documentation. https://github.com/ARCLab-MIT/pyssem/

        Args:
            MOCAT_config (json): Dictionary containing the MOCAT configuration parameters.

        Returns:
            Model: An configured instance of the MOCAT model.
    """
    scenario_props = MOCAT_config["scenario_properties"]
    try:
        economic_parameters_json = MOCAT_config['OPUS']
    except:
        raise ValueError("Please ensure that the economic parameters of OPUS are configured correctly.")

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
    )

    species = MOCAT_config["species"]

    MOCAT.configure_species(species)
    MOCAT.opus_active_loss_setup(fringe_satellite)
    MOCAT.initial_population() 
    MOCAT.build_model()

    print("You have these species in the model: ", MOCAT.scenario_properties.species_names)

    # Find the PMD linked species and return the index. 
    pmd_linked_species_to_fringe = [
        species
        for species_group in MOCAT.scenario_properties.species.values()
        for species in species_group
        if any(linked_species.sym_name == fringe_satellite for linked_species in species.pmd_linked_species)
    ]

    if len(pmd_linked_species_to_fringe) != 1:
        raise ValueError("Please ensure that there is only one species linked to the fringe satellite.")
    else:
        pmd_linked_species = pmd_linked_species_to_fringe[0].sym_name
 
    return MOCAT, economic_parameters_json, pmd_linked_species