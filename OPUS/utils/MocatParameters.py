from ast import If
from pyssem.model import Model
from sympy.series.gruntz import I # mocat-ssem
from .MultiSpecies import MultiSpecies
from .EconParameters import EconParameters
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable, Mapping

def configure_mocat(MOCAT_config: json, multi_species: MultiSpecies = None, grid_search: bool = False) -> Model:
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
        eccentricity_bins=scenario_props.get("eccentricity_bins", None),
        opus=True
    )

    species = MOCAT_config["species"]

    MOCAT.configure_species(species)
    # Create an active_loss_setup for each of the species in the model.
    if multi_species != None:
        # Commented out opus_collisions_setup as it doesn't exist in current pyssem version
        for species in multi_species.species:
            # get maneuverability from species, where ['active'] list has sym_name of the species
            mocat_species = next((mocat_species for mocat_species in MOCAT.scenario_properties.species['active'] if mocat_species.sym_name == species.name), None)
            if mocat_species.maneuverable:
                MOCAT.opus_collisions_setup(species.name, maneuvers=True)
                species.maneuverable = True
            else:
                MOCAT.opus_collisions_setup(species.name, maneuvers=False)
                species.maneuverable = False
        pass

    MOCAT.build_model(elliptical=scenario_props.get("elliptical", False))

    print("You have these species in the model: ", MOCAT.scenario_properties.species_names)

    # Find the PMD linked species and return the index. 
    if multi_species == None:
        return MOCAT

    for opus_species in multi_species.species:
        pmd_linked_species_to_fringe = []
        
        for species_group in MOCAT.scenario_properties.species.values():
            for species in species_group:
                # Check if species has pmd_linked_species attribute and it's not None
                if hasattr(species, 'pmd_linked_species') and species.pmd_linked_species is not None:
                    # Check if any linked species matches the opus_species name
                    # check if pmd_linked_species is a list
                    if isinstance(species.pmd_linked_species, list):
                        for linked_species in species.pmd_linked_species:
                            if linked_species.sym_name == opus_species.name:
                                pmd_linked_species_to_fringe.append(species)
                                break  # Found a match, no need to check other linked_species
                    else:
                        if species.pmd_linked_species.sym_name == opus_species.name:
                            pmd_linked_species_to_fringe.append(species)
                            break  # Found a match, no need to check other linked_species

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



def _compute_shell_edges(shell_centers: np.ndarray) -> np.ndarray:
    """Derive shell edges from the nominal altitude grid."""
    shell_centers = np.asarray(shell_centers, dtype=float)
    if shell_centers.ndim != 1:
        raise ValueError("Shell centers should be a 1-D array.")
    if shell_centers.size == 1:
        half_width = 50.0
        return np.array([shell_centers[0] - half_width, shell_centers[0] + half_width])
    deltas = np.diff(shell_centers)
    lower_edge = shell_centers[0] - deltas[0] / 2.0
    upper_edge = shell_centers[-1] + deltas[-1] / 2.0
    inner_edges = shell_centers[:-1] + deltas / 2.0
    return np.concatenate(([lower_edge], inner_edges, [upper_edge]))


def _mean_altitude_from_row(row: Mapping[str, object]) -> float | None:
    """Estimate altitude using perigee/apogee measurements."""
    perigee = pd.to_numeric(row.get("Perigee (km)"), errors="coerce")
    apogee = pd.to_numeric(row.get("Apogee (km)"), errors="coerce")
    values = [val for val in (perigee, apogee) if pd.notna(val)]
    if not values:
        return None
    if len(values) == 2:
        return float(sum(values) / 2.0)
    return float(values[0])


def override_initial_population_from_classified_csv(
    MOCAT: Model,
    csv_path: str | Path,
    species_of_interest: Iterable[str] = ("S", "Su", "Sns"),
    altitude_column: str = "mean_altitude_km",
    before_date: str | pd.Timestamp | None = None,
) -> None:
    """
    Replace the initial population (x0) with counts derived from a classified catalogue.

    Parameters
    ----------
    MOCAT : Model
        Configured MOCAT model whose initial population will be overwritten in-place.
    csv_path : str | Path
        Path to the classified CSV (e.g., produced by classify_satellites.py).
    species_of_interest : Iterable[str], optional
        Iterable of species names that should be overwritten.
    altitude_column : str, optional
        Column containing altitude estimates in kilometres. If absent, the column
        will be synthesised via the perigee/apogee fields.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Classified catalogue not found at {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig", engine="python")

    if altitude_column not in df.columns:
        df[altitude_column] = df.apply(_mean_altitude_from_row, axis=1)
    else:
        df[altitude_column] = pd.to_numeric(df[altitude_column], errors="coerce")

    df["Date of Launch"] = pd.to_datetime(df.get("Date of Launch"), errors="coerce")

    df = df[df["species_class"].isin(species_of_interest)].copy()

    if df.empty:
        print("Warning: No satellite records found for requested species; x0 unchanged.")
        return

    if before_date is not None:
        cutoff = pd.to_datetime(before_date)
        df = df[df["Date of Launch"] < cutoff]
        if df.empty:
            print(
                f"Warning: No satellite records found before {cutoff.date()}; "
                "requested species left unchanged."
            )
            return

    # Fill missing altitude values with species-specific medians (fallback to global median).
    available_altitudes = df[altitude_column].dropna()
    global_median_alt = available_altitudes.median() if not available_altitudes.empty else None

    species_medians = (
        df.groupby("species_class")[altitude_column]
        .median(numeric_only=True)
        .to_dict()
    )

    def _fill_altitude(row):
        value = row[altitude_column]
        if pd.notna(value):
            return value
        species = row["species_class"]
        species_median = species_medians.get(species)
        if species_median is not None and not np.isnan(species_median):
            return species_median
        return global_median_alt

    df[altitude_column] = df.apply(_fill_altitude, axis=1)
    df = df[df[altitude_column].notna()]

    species_names = list(MOCAT.scenario_properties.species_names)
    x0 = MOCAT.scenario_properties.x0

    if x0.ndim == 1:
        num_species = len(species_names)
        if num_species == 0:
            raise ValueError("Cannot infer shell count: no species defined.")
        n_shells = len(x0) // num_species
    elif x0.ndim >= 2:
        n_shells = x0.shape[0]
    else:
        raise ValueError(f"Unsupported x0 dimensionality: {x0.ndim}")

    if n_shells <= 0:
        raise ValueError("Inferred zero shells when overriding x0.")

    if df.empty:
        print("Warning: No valid satellite records found to override x0.")
        if isinstance(x0, pd.DataFrame):
            for species in species_of_interest:
                if species in x0.columns:
                    x0.loc[:, species] = 0
        else:
            for species in species_of_interest:
                try:
                    species_idx = species_names.index(species)
                except ValueError:
                    continue
                if x0.ndim == 1:
                    start = species_idx * n_shells
                    end = start + n_shells
                    x0[start:end] = 0
                elif x0.ndim == 2:
                    x0[:, species_idx] = 0
                elif x0.ndim == 3:
                    x0[:, species_idx, :] = 0
        return

    shell_axis = np.asarray(MOCAT.scenario_properties.R0_km, dtype=float)

    if shell_axis.ndim != 1:
        raise ValueError("MOCAT scenario R0_km must be a 1-D array.")
    if shell_axis.size == n_shells + 1:
        shell_edges = shell_axis.astype(float)
    elif shell_axis.size == n_shells:
        shell_edges = _compute_shell_edges(shell_axis)
    else:
        raise ValueError(
            f"Unexpected number of altitude reference points: {shell_axis.size} for {n_shells} shells."
        )

    shell_indices = np.digitize(df[altitude_column].to_numpy(), shell_edges) - 1
    shell_indices = np.clip(shell_indices, 0, n_shells - 1)
    df["_shell_idx"] = shell_indices

    species_to_counts = {
        species: np.zeros(n_shells, dtype=float) for species in species_of_interest
    }

    for species, group in df.groupby("species_class"):
        if species not in species_to_counts:
            continue
        idx, freq = np.unique(group["_shell_idx"].to_numpy(), return_counts=True)
        species_to_counts[species][idx] = freq

    is_dataframe = isinstance(x0, pd.DataFrame)
    x0_dtype = getattr(x0, "dtype", float) if not is_dataframe else float

    for species, counts in species_to_counts.items():
        try:
            species_idx = species_names.index(species)
        except ValueError:
            print(f"Warning: Species '{species}' not found in MOCAT scenario; skipping.")
            continue
        counts = counts.astype(x0_dtype, copy=False)
        if is_dataframe:
            if species in x0.columns:
                x0.loc[:, species] = counts
            else:
                print(f"Warning: Species '{species}' column missing from x0 DataFrame; skipping.")
            continue
        if x0.ndim == 1:
            start = species_idx * n_shells
            end = start + n_shells
            x0[start:end] = counts
        elif x0.ndim == 2:
            x0[:, species_idx] = counts
        elif x0.ndim == 3:
            x0[:, species_idx, :] = 0
            x0[:, species_idx, 0] = counts
        else:
            raise ValueError(f"Unsupported x0 dimensionality: {x0.ndim}")
            
    if "_shell_idx" in df.columns:
        df.drop(columns="_shell_idx", inplace=True)

    return MOCAT