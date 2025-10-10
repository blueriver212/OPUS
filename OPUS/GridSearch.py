# import json
# import numpy as np
# from itertools import product
# from copy import deepcopy
# import contextlib
# import io
# from tqdm import tqdm
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from main import IAMSolver

# TARGET_COUNTS = {"S": 6300, "Su": 2280, "Sns": 801}
# INTERCEPT_GRID_S = np.linspace(6.5e5, 8.5e5, 5) # should be around 7.5e5
# INTERCEPT_GRID_SU = np.linspace(6.5e5, 7.5e5, 5) # should be around 5.5e5
# INTERCEPT_GRID_SNS = np.linspace(0.9e4, 1.5e5, 5) # should be around 1e5,

# def get_total_species_from_output(species_data):
#     totals = {}
#     for species, data_array in species_data.items():
#         if isinstance(data_array, np.ndarray):
#             totals[species] = np.sum(data_array[-1])
#     return totals

# def run_simulation(intercepts):
#     with open("./OPUS/configuration/multi_single_species.json") as f:
#         base_config = json.load(f)

#     config = deepcopy(base_config)
#     for species in config["species"]:
#         name = species["sym_name"]
#         if name in intercepts:
#             species["OPUS"]["intercept"] = intercepts[name]

#     sim_name = "RevenueInterceptSearch"
#     scenario_name = "Baseline"
#     iam_solver = IAMSolver()

#     with contextlib.redirect_stdout(io.StringIO()):
#         species_data = iam_solver.iam_solver(scenario_name, config, sim_name, grid_search=True)

#     return get_total_species_from_output(species_data)

# def compute_cost(result):
#     return sum((result[sp] - TARGET_COUNTS[sp])**2 for sp in TARGET_COUNTS)

# def evaluate_combination(intercepts_tuple):
#     s_val, su_val, sns_val = intercepts_tuple
#     intercept_dict = {"S": s_val, "Su": su_val, "Sns": sns_val}
#     result = run_simulation(intercept_dict)
#     cost = compute_cost(result)
#     return intercept_dict, result, cost

# def main():
#     combinations = list(product(INTERCEPT_GRID_S, INTERCEPT_GRID_SU, INTERCEPT_GRID_SNS))
#     total_combinations = len(combinations)

#     best_cost = float("inf")
#     best_intercepts = None
#     results_log = []

#     with ProcessPoolExecutor() as executor:
#         futures = {executor.submit(evaluate_combination, comb): comb for comb in combinations}
#         for future in tqdm(as_completed(futures), total=total_combinations, desc="Parallel Grid Search"):
#             intercept_dict, result, cost = future.result()
#             results_log.append((intercept_dict, result, cost))

#             print(f"Intercepts: {intercept_dict}, Final: {result}, Cost: {cost:.2f}")
#             if cost < best_cost:
#                 best_cost = cost
#                 best_intercepts = intercept_dict
#             if cost == 0:
#                 break

#     print(f"\n✅ Best intercepts found: {best_intercepts} with cost {best_cost:.2f}")

# if __name__ == "__main__":
#     from multiprocessing import freeze_support
#     freeze_support()
#     main()

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


# ───────────────────────────────────────────────────
# 0.  CONFIGURATION
# ───────────────────────────────────────────────────
N_CALLS          = 60          # <— total model evaluations
N_INITIAL_POINTS = 10
RANDOM_STATE     = 42
SHOW_SURFACE     = True        # set False if you only need the optimiser

TARGET_COUNTS = {"S": 6300, "Su": 2280, "Sns": 801}


# ───────────────────────────────────────────────────
# 1.  HELPER ROUTINES (unchanged logic)
# ───────────────────────────────────────────────────
def get_total_species_from_output(species_data):
    totals = {}
    for species, data_array in species_data.items():
        if isinstance(data_array, np.ndarray):
            totals[species] = np.sum(data_array[-1])
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


# # ───────────────────────────────────────────────────
# # 2.  BAYESIAN OPTIMISATION
# # ───────────────────────────────────────────────────
# # search_space = [
# #     Real(6.0e5, 9.0e5, name="S"),
# #     Real(5.0e5, 8.0e5, name="Su"),
# #     Real(0.0e0, 2.0e5, name="Sns"),
# # ]

# search_space = [
#     Real(9.0e5, 1.1e6, name="S"),
#     Real(8.0e5, 1.0e6, name="Su"),
#     Real(0.5e5, 2.0e5, name="Sns"),   # leave generous for now
# ]

# @use_named_args(search_space)
# def objective(S, Su, Sns):
#     """Objective wrapper for gp_minimize."""
#     t0 = time.perf_counter()
#     cost = compute_cost(run_simulation({"S": S, "Su": Su, "Sns": Sns}))
#     dt  = time.perf_counter() - t0
#     # update progress bar (defined outside) with timing info
#     pbar.update(1)
#     pbar.set_postfix(cost=f"{cost:,.0f}", last=f"{dt:4.1f}s")
#     return cost


# # ── progress bar set-up ─────────────────────────────────
# pbar = tqdm(total=N_CALLS, desc="Bayesian optimising", ncols=80)

# start_all = time.perf_counter()
# result = gp_minimize(
#     func             = objective,
#     dimensions       = search_space,
#     n_calls          = N_CALLS,
#     n_initial_points = N_INITIAL_POINTS,
#     acq_func         = "EI",
#     x0 = [1e6,   # clip S
#       1e6, # clip Su
#       1.5e5],            # Sns already within [0.5e5, 2.0e5]
#     random_state     = RANDOM_STATE,
#     n_jobs           = -1
# )
# pbar.close()
# print(f"\n🏁  Finished in {(time.perf_counter() - start_all)/60:4.1f} min")

# best_S, best_Su, best_Sns = result.x
# print(f"Best intercepts:  S={best_S:.1f},  Su={best_Su:.1f},  "
#       f"Sns={best_Sns:.1f}   Cost={result.fun:,.0f}")

# ───────────────────────────────────────────────────
# 2.  JACOBIAN / LEAST-SQUARES CALIBRATION
#     (≈ 5–8 simulator runs total)
# ───────────────────────────────────────────────────

import numpy as np
from tqdm import tqdm

# --- helper ----------------------------------------------------------
def sim_counts(R_vec):
    """vectorised wrapper → np.array([N_S, N_Su, N_Sns])"""
    names = ["S", "Su", "Sns"]
    result = run_simulation(dict(zip(names, R_vec)))
    return np.array([result[n] for n in names])

def jacobian(R_base, delta=30_000):
    """Finite-difference Jacobian  ∂N_i / ∂R_j  (3×3)."""
    J = np.zeros((3, 3))
    N0 = sim_counts(R_base)
    for j in range(3):
        R_pert       = R_base.copy()
        R_pert[j]   += delta
        Nj           = sim_counts(R_pert)
        J[:, j]      = (Nj - N0) / delta
    return J, N0

# --- main routine ----------------------------------------------------
TARGET = np.array([6300, 2280, 801])
N_PASSES      = 2        # 1 pass is often enough; 2 gives “tight” fit
DELTA         = 30_000   # finite-difference step (adjust if sensitivity is tiny)

# starting guess – use last BO best if you have it, otherwise mid-box
R = np.array([1_000_000, 900_000, 120_000], dtype=float)

for it in range(N_PASSES):
    print(f"\nPass {it+1}")
    J, N0 = jacobian(R, delta=DELTA)

    # Solve  J ΔR = (T – N)   → least-squares in case J is not perfectly diagonal
    dR, *_ = np.linalg.lstsq(J, TARGET - N0, rcond=None)
    R      = R + dR

    # Evaluate new intercepts once
    N1  = sim_counts(R)
    cost = compute_cost(dict(zip(["S","Su","Sns"], N1)))

    print("  Jacobian:\n", np.round(J, 2))
    print("  ΔR:", np.round(dR).astype(int))
    print("  New intercepts:", np.round(R).astype(int))
    print("  Counts:", np.round(N1).astype(int), f"  Cost={cost:,.0f}")

    # simple early stop
    if cost < 1e5:          # RMS error ≈ 316
        break

print("\n✅  Final intercepts  → ", np.round(R).astype(int))

# ───────────────────────────────────────────────────
# 3.  3-D SURFACE VISUALISATION  (optional)
# ───────────────────────────────────────────────────
def plot_cost_surface(fixed_Sns=None, num_pts=30):
    """
    Fix Sns (e.g. at optimum) and show cost surface over S × Su.
    """
    if fixed_Sns is None:
        fixed_Sns = best_Sns

    S_vals  = np.linspace(6.5e5, 9.0e5, num_pts)
    Su_vals = np.linspace(5.5e5, 8.0e5, num_pts)
    S_grid, Su_grid = np.meshgrid(S_vals, Su_vals)

    Z = np.empty_like(S_grid, dtype=float)
    outer = tqdm(range(num_pts), desc="Surface rows", ncols=80)
    for i in outer:
        for j in range(num_pts):
            res = run_simulation(
                {"S": S_grid[i, j], "Su": Su_grid[i, j], "Sns": fixed_Sns}
            )
            Z[i, j] = compute_cost(res)

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot_surface(S_grid, Su_grid, Z, alpha=0.75, linewidth=0)
    ax.set_xlabel("Intercept S")
    ax.set_ylabel("Intercept Su")
    ax.set_zlabel("Cost")
    ax.set_title(f"Cost surface at Sns = {fixed_Sns:,.0f}")
    plt.tight_layout()
    plt.show()


# if SHOW_SURFACE:
#     plot_cost_surface(fixed_Sns=best_Sns, num_pts=30)