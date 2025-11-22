"""
bayes_opt_intercepts.py
----------------------------------------------------
Bayesian optimisation of the IAMSolver intercepts
using scikit-optimize’s Gaussian-process engine
(gp_minimize) + an optional 3-D surface plot.

Requirements
------------
pip install scikit-optimize matplotlib tqdm
"""

"""
bayes_opt_intercepts_progress.py
--------------------------------
Like the previous script but with live progress bars and timing.
"""
# Set start date in json to 2017 and set lifetime to 8 years
#Run with OPUS not OPUSB
import json, io, contextlib, time
from copy import deepcopy
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D      # noqa: F401
from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from main import IAMSolver
from concurrent.futures import ProcessPoolExecutor


# ───────────────────────────────────────────────────
# 0.  CONFIGURATION
# ───────────────────────────────────────────────────
N_CALLS          = 60          # <— total model evaluations
N_INITIAL_POINTS = 10
RANDOM_STATE     = 42
SHOW_SURFACE     = True        # set False if you only need the optimiser

TARGET_COUNTS = {"S": 7677, "Su": 2665, "Sns": 1228}
# TARGET_COUNTS = {"SA": 3070, "SB": 3070, "SC": 1537, "SuA": 1075, "SuB": 1075, "SuC": 540}


# ───────────────────────────────────────────────────
# 1.  HELPER ROUTINES (unchanged logic)
# ───────────────────────────────────────────────────
def get_total_species_from_output(species_data):
        totals = {}
        for species, year_data in species_data.items():
            if isinstance(year_data, dict):
                # Get the latest year's data
                latest_year = max(year_data.keys())
                latest_data = year_data[latest_year]
                
                if isinstance(latest_data, np.ndarray):
                    # Sum the array values
                    totals[species] = np.sum(latest_data)
                elif hasattr(latest_data, 'sum'):
                    # Handle pandas Series
                    totals[species] = latest_data.sum()
                else:
                    # Fallback for other data types
                    totals[species] = float(latest_data) if isinstance(latest_data, (int, float)) else 0
            elif isinstance(year_data, np.ndarray):
                # Handle direct array input (backward compatibility)
                totals[species] = np.sum(year_data[-1])
        
        return totals


def run_simulation(intercepts):
    """
    Run IAMSolver once with
      • species-specific revenue intercepts  (passed in `intercepts`)
      • a coefficient that is *derived* from that revenue:
            coefficient = revenue / (2 · target_end_sats)

    Returns a dict  {species → final total}.
    """
    # ── read the template JSON once and cache it ──────────────────
    if not hasattr(run_simulation, "_baseline"):
        with open("./OPUS/configuration/multi_single_species.json") as f:
            run_simulation._baseline = json.load(f)

    config = deepcopy(run_simulation._baseline)

    for spec in config["species"]:
        name = spec["sym_name"]
        if name in intercepts:
            revenue = intercepts[name]                         # ← chosen by BO
            spec["OPUS"]["intercept"] = revenue

            # ---------- NEW: revenue-driven coefficient ----------
            # goal = target end-sats for that species
            goal   = TARGET_COUNTS[name]
            coeff  = revenue / (2 * goal)                      # per problem statement
            spec["OPUS"]["coefficient"] = coeff               # make sure this field exists in your JSON
            # -----------------------------------------------------

    iam_solver   = IAMSolver()
    sim_name     = "RevenueInterceptSearch"
    scenario     = "Baseline"

    with contextlib.redirect_stdout(io.StringIO()):
        species_data = iam_solver.iam_solver(
            scenario, config, sim_name, grid_search=True
        )

    return get_total_species_from_output(species_data)

def compute_cost(result):
    """Sum-squared error vs. TARGET_COUNTS."""
    return sum((result[sp] - TARGET_COUNTS[sp])**2 for sp in TARGET_COUNTS)


import numpy as np
from tqdm import tqdm

# --- helper ----------------------------------------------------------
def sim_counts(R_vec):
    """vectorised wrapper → np.array([N_S, N_Su, N_Sns])"""
    names = ["S", "Su", "Sns"]
    # names = ["SA", "SB", "SC", "SuA", "SuB", "SuC"]
    result = run_simulation(dict(zip(names, R_vec)))
    return np.array([result[n] for n in names])

def jacobian(R_base, delta=30_000):
    """Parallel finite-difference Jacobian  ∂N_i / ∂R_j  (3×3)."""
    J = np.zeros((3, 3))
    
    # 1. Calculate baseline (can't parallelize this easily against the perturbations)
    N0 = sim_counts(R_base)

    # 2. Prepare the inputs for the 3 parallel simulations
    perturbations = []
    for j in range(3):
        R_pert = R_base.copy()
        R_pert[j] += delta
        perturbations.append(R_pert)

    # 3. Run the 3 perturbations in parallel
    # This cuts the wait time for this step by ~66%
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(sim_counts, perturbations))

    # 4. Construct the Jacobian Matrix from results
    for j, Nj in enumerate(results):
        J[:, j] = (Nj - N0) / delta

    return J, N0

# --- main routine ----------------------------------------------------
if __name__ == "__main__":
    TARGET = np.array([7677, 2665, 1228])
    # TARGET = np.array([3070, 3070, 1537, 1075, 1075, 540])
    N_PASSES      = 4        # 1 pass is often enough; 2 gives “tight” fit
    DELTA         = 30_000   # finite-difference step

    # starting guess – use last BO best if you have it, otherwise mid-box
    # R = np.array([1629368, 1629368, 1629368, 2092593, 2092593, 2092593], dtype=float)
    R = np.array([1665764, 2205401, 55701], dtype=float)

    for it in range(N_PASSES):
        print(f"\nPass {it+1}")
        # This now calls the parallel version
        J, N0 = jacobian(R, delta=DELTA)

        # Solve  J ΔR = (T – N)
        dR, *_ = np.linalg.lstsq(J, TARGET - N0, rcond=None)
        R      = R + dR

        # Evaluate new intercepts once (strictly speaking, you could perform this check 
        # inside the next loop iteration to save 1 sim, but this is safer for reporting)
        N1  = sim_counts(R)
        cost = compute_cost(dict(zip(["S","Su","Sns"], N1)))
        
        print("  Jacobian:\n", np.round(J, 2))
        print("  ΔR:", np.round(dR).astype(int))
        print("  New intercepts:", np.round(R).astype(int))
        print("  Counts:", np.round(N1).astype(int), f"  Cost={cost:,.0f}")

        # simple early stop
        if cost < 1e5:          
            break

print("\n✅  Final intercepts  → ", np.round(R).astype(int))