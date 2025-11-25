"""
bayes_opt_intercepts.py
----------------------------------------------------
Bayesian optimisation of the IAMSolver intercepts
using Broyden's method for Jacobian updates to save time.
"""

import json, io, contextlib, time
from copy import deepcopy
from itertools import product

import numpy as np
# import matplotlib.pyplot as plt # Not strictly needed for this script
# from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from main import IAMSolver
from concurrent.futures import ProcessPoolExecutor


# ───────────────────────────────────────────────────
# 0.  CONFIGURATION
# ───────────────────────────────────────────────────
N_CALLS          = 60
N_INITIAL_POINTS = 10
RANDOM_STATE     = 42
SHOW_SURFACE     = True

TARGET_COUNTS = {"S": 7677, "Su": 2665, "Sns": 1228}
# TARGET_COUNTS = {"SA": 3070, "SB": 3070, "SC": 1537, "SuA": 1075, "SuB": 1075, "SuC": 540}

# ───────────────────────────────────────────────────
# 1.  HELPER ROUTINES
# ───────────────────────────────────────────────────
def get_total_species_from_output(species_data):
    totals = {}
    for species, year_data in species_data.items():
        if isinstance(year_data, dict):
            latest_year = max(year_data.keys())
            latest_data = year_data[latest_year]
            
            if isinstance(latest_data, np.ndarray):
                totals[species] = np.sum(latest_data)
            elif hasattr(latest_data, 'sum'):
                totals[species] = latest_data.sum()
            else:
                totals[species] = float(latest_data) if isinstance(latest_data, (int, float)) else 0
        elif isinstance(year_data, np.ndarray):
            totals[species] = np.sum(year_data[-1])
    
    return totals


def run_simulation(intercepts):
    if not hasattr(run_simulation, "_baseline"):
        # UPDATE THIS PATH IF NEEDED
        with open("./OPUS/configuration/multi_single_species.json") as f:
            run_simulation._baseline = json.load(f)

    config = deepcopy(run_simulation._baseline)

    for spec in config["species"]:
        name = spec["sym_name"]
        if name in intercepts:
            revenue = intercepts[name]
            spec["OPUS"]["intercept"] = revenue

            goal   = TARGET_COUNTS[name]
            coeff  = revenue / (2 * goal)
            spec["OPUS"]["coefficient"] = coeff

    iam_solver   = IAMSolver()
    sim_name     = "RevenueInterceptSearch"
    scenario     = "Baseline"

    with contextlib.redirect_stdout(io.StringIO()):
        species_data = iam_solver.iam_solver(
            scenario, config, sim_name, grid_search=True
        )

    return get_total_species_from_output(species_data)

def compute_cost(result):
    return sum((result[sp] - TARGET_COUNTS[sp])**2 for sp in TARGET_COUNTS)


# --- helper ----------------------------------------------------------
def sim_counts(R_vec):
    names = ["S", "Su", "Sns"]
    result = run_simulation(dict(zip(names, R_vec)))
    return np.array([result[n] for n in names])

def jacobian(R_base, delta=30_000):
    J = np.zeros((3, 3))
    
    # 1. Calculate baseline
    N0 = sim_counts(R_base)

    # 2. Prepare inputs
    perturbations = []
    for j in range(3):
        R_pert = R_base.copy()
        R_pert[j] += delta
        perturbations.append(R_pert)

    # 3. Run parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(sim_counts, perturbations))

    # 4. Construct Jacobian
    for j, Nj in enumerate(results):
        J[:, j] = (Nj - N0) / delta

    return J, N0

# ───────────────────────────────────────────────────
# MAIN ROUTINE WITH CHECKPOINTING
# ───────────────────────────────────────────────────
if __name__ == "__main__":
    TARGET = np.array([7677, 2665, 1228])
    N_PASSES      = 4
    DELTA         = 30_000

    # STARTING GUESS
    # R = np.array([1665764, 2205401, 55701], dtype=float)
    R = np.array([1665764, 2205401, 55701], dtype=float)

    # Variables for Broyden history
    J      = None  
    N_prev = None  
    R_prev = None  
    
    print(f"--- Starting Optimisation (Broyden Mode + AutoSave) ---")
    print(f"Initial Intercepts: {R}")

    for it in range(N_PASSES):
        print(f"\n=== PASS {it+1} / {N_PASSES} ===")
        start_time = time.time()
        
        # -----------------------------------------------------------
        # STEP 1: Get Baseline (Takes ~4 hours)
        # -----------------------------------------------------------
        if it == 0:
            print("  > Running baseline simulation...")
            N0 = sim_counts(R)
        else:
            # Use the verification result from the end of the last loop
            N0 = N1 
            
        cost = compute_cost(dict(zip(["S","Su","Sns"], N0)))
        print(f"  Current Counts: {np.round(N0).astype(int)} Cost: {cost:,.0f}")

        if cost < 1e5:
            print("  > Converged!")
            break

        # --- SAVE POINT 1: START OF PASS ---
        try:
            with open("intercept_log.txt", "a") as log_file:
                log_file.write(f"\nPASS {it+1} START | Cost: {cost:.0f}\n")
                log_file.write(f"Current R: {np.round(R).astype(int).tolist()}\n")
                log_file.write("-" * 30 + "\n")
        except Exception as e:
            print(f"  > [Warning] Could not save log: {e}")

        # -----------------------------------------------------------
        # STEP 2: Get Jacobian (Full or Broyden)
        # -----------------------------------------------------------
        if it == 0:
            print("  > Calculating FULL Jacobian (Running parallel workers)...")
            perturbations = []
            for j in range(3):
                R_pert = R.copy()
                R_pert[j] += DELTA
                perturbations.append(R_pert)
            
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(sim_counts, perturbations))
            
            J = np.zeros((3, 3))
            for j, Nj in enumerate(results):
                J[:, j] = (Nj - N0) / DELTA
            print("  > Full Jacobian calculated.")
                
        else:
            print("  > Approximating Jacobian (Broyden update)...")
            dN = N0 - N_prev
            dR_step = R - R_prev
            
            # Broyden formula
            numer = dN - J @ dR_step
            denom = np.dot(dR_step, dR_step)
            
            # Safety check for divide by zero
            if denom < 1e-9:
                 print("  > Step too small, skipping Jacobian update.")
            else:
                 J = J + np.outer(numer, dR_step) / denom

        # -----------------------------------------------------------
        # STEP 3: Solve and Step
        # -----------------------------------------------------------
        # Save state for next Broyden update
        R_prev = R.copy()
        N_prev = N0.copy()

        # Standard Newton Step
        dR_solve, *_ = np.linalg.lstsq(J, TARGET - N0, rcond=None)
        
        # Apply Damping (optional, keeps it stable)
        damping = 1.0
        R = R + (dR_solve * damping)
        
        # Prevent negative intercepts
        R[R < 0] = 1000
        
        print("  Jacobian:\n", np.round(J, 2))
        print("  ΔR:", np.round(dR_solve).astype(int))
        print("  Next Intercept Guess:", np.round(R).astype(int))

        # -----------------------------------------------------------
        # STEP 4: Verification Run
        # -----------------------------------------------------------
        print("  > Verifying new position...")
        N1 = sim_counts(R) # This becomes N0 in the next loop
        new_cost = compute_cost(dict(zip(["S","Su","Sns"], N1)))
        
        elapsed = (time.time() - start_time) / 3600
        print(f"  Pass Complete in {elapsed:.2f} hours. New Cost: {new_cost:,.0f}")

        # --- SAVE POINT 2: END OF PASS ---
        try:
            with open("intercept_log.txt", "a") as log_file:
                log_file.write(f"PASS {it+1} END | Time: {elapsed:.2f}h | Cost: {new_cost:.0f}\n")
                log_file.write(f"Jacobian:\n{np.round(J, 2)}\n")
                log_file.write(f"Proposed R: {np.round(R).astype(int).tolist()}\n")
                log_file.write(f"Resulting N: {np.round(N1).astype(int).tolist()}\n")
                log_file.write("=" * 30 + "\n")
            print("  > Log saved to 'intercept_log.txt'")
        except Exception as e:
            print(f"  > [Warning] Could not save log: {e}")

    # Indented print to prevent multiprocessing crash
    print("\n✅  Final intercepts  → ", np.round(R).astype(int))