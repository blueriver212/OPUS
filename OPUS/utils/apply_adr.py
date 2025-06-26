import numpy as np

def apply_ADR(state_vec: np.ndarray, mocat, operations: list):
    """
    Apply a list of removal operations to state_vec and return a copy.

    Parameters
    ----------
    state_vec   : flat 1-D state vector (len = n_shells * n_species_total)
    mocat       : pyssem.model.Model instance (for species ordering)
    operations  : iterable of dicts having keys
                  { "species": str, "shell": int, "num_remove": int }
    """
    n_shells  = mocat.scenario_properties.n_shells
    new_state = state_vec.copy()

    # ------------------------------------------------------------------
    # Build offsets from *species_names* – guaranteed correct ordering
    # ------------------------------------------------------------------
    offsets = {
        sym_name: (i * n_shells, (i + 1) * n_shells)
        for i, sym_name in enumerate(mocat.scenario_properties.species_names)
    }

    # ------------------------------------------------------------------
    # Apply each operation
    # ------------------------------------------------------------------
    for op in operations:
        sym   = op["species"].strip()            # guard against stray spaces
        k     = int(op["shell"])
        n_rem = int(op["num_remove"])

        if sym not in offsets:
            raise KeyError(f"Species '{sym}' not in model.")
        if not (0 <= k < n_shells):
            raise IndexError(f"Shell {k} out of range 0–{n_shells-1}.")

        flat_idx = offsets[sym][0] + k
        new_state[flat_idx] = max(0, new_state[flat_idx] - n_rem)

    return new_state