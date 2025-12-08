import json, io, contextlib, csv, os
import numpy as np
from copy import deepcopy
from scipy.optimize import least_squares
from main import IAMSolver

# ───────────────────────────────────────────────────
# 0. CONFIGURATION
# ───────────────────────────────────────────────────
# Target order must match the species list: [SA, SB, SC, SuA, SuB, SuC]
TARGET_VEC = np.array([3070, 3070, 1537, 1075, 1075, 540])
SPECIES_ORDER = ["SA", "SB", "SC", "SuA", "SuB", "SuC"]
LOG_FILE = "calibration_log_robust.csv"

# ───────────────────────────────────────────────────
# 1. HELPER ROUTINES
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
                totals[species] = float(latest_data)
        elif isinstance(year_data, np.ndarray):
            totals[species] = np.sum(year_data[-1])
    return totals

def run_simulation(intercepts):
    # Cache baseline to speed up loads
    if not hasattr(run_simulation, "_baseline"):
        with open("./OPUS/configuration/bonded_species.json") as f:
            run_simulation._baseline = json.load(f)

    config = deepcopy(run_simulation._baseline)
    
    for spec in config["species"]:
        name = spec["sym_name"]
        if name in intercepts:
            revenue = intercepts[name]
            spec["OPUS"]["intercept"] = revenue
            # Update coefficient: Revenue = Coeff * 2 * Goal -> Coeff = Rev / (2*Goal)
            # (Goal is fixed to target)
            goal_count = dict(zip(SPECIES_ORDER, TARGET_VEC)).get(name, 1000)
            spec["OPUS"]["coefficient"] = revenue / (2 * goal_count)
            
        if "OPUS" in spec:
            spec["OPUS"]["bond"] = None

    iam_solver = IAMSolver()
    if hasattr(iam_solver, 'multi_species_names'):
        iam_solver.multi_species_names = SPECIES_ORDER
    if hasattr(iam_solver, 'bonded_species_names'):
        iam_solver.bonded_species_names = []

    # Suppress internal prints
    with contextlib.redirect_stdout(io.StringIO()):
        species_data = iam_solver.iam_solver(
            "Baseline", config, "Calib", 
            multi_species_names=SPECIES_ORDER, grid_search=True
        )
    return get_total_species_from_output(species_data)

def log_result(R_vec, N_vec, residuals):
    """Saves every single evaluation to CSV."""
    file_exists = os.path.isfile(LOG_FILE)
    cost = np.sum(residuals**2)
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ['R_' + s for s in SPECIES_ORDER] + \
                     ['N_' + s for s in SPECIES_ORDER] + ['Cost']
            writer.writerow(header)
        writer.writerow(list(R_vec) + list(N_vec) + [cost])

# ───────────────────────────────────────────────────
# 2. OPTIMIZATION FUNCTION
# ───────────────────────────────────────────────────
def objective_function(R_vec):
    """
    Returns the vector of residuals (Predicted - Target).
    least_squares tries to minimize sum(residuals^2).
    """
    # 1. Run Sim
    intercepts = dict(zip(SPECIES_ORDER, R_vec))
    results = run_simulation(intercepts)
    
    # 2. Extract Counts in correct order
    N_vec = np.array([results.get(s, 0) for s in SPECIES_ORDER])
    
    # 3. Calculate Residuals (N - Target)
    residuals = N_vec - TARGET_VEC
    
    # 4. Log & Print
    cost = np.sum(residuals**2)
    print(f"[Sim] R: {np.round(R_vec).astype(int)} | N: {np.round(N_vec).astype(int)} | Cost: {cost:,.0f}")
    log_result(R_vec, N_vec, residuals)
    
    # Optional: Penalize negative counts heavily (though bounds handle R, physics handles N)
    return residuals

# ───────────────────────────────────────────────────
# 3. MAIN
# ───────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Starting Robust Calibration (Trust Region Reflective)...")
    print(f"Targets: {dict(zip(SPECIES_ORDER, TARGET_VEC))}\n")

    # Initial Guess (Your previous R that was kind of working, or a safe middle ground)
    # Using the values from your earlier script as a safer start than the divergent ones
    x0 = np.array([1629368, 1629368, 1629368, 2092593, 2092593, 2092593], dtype=float)

    # Bounds: Revenue Intercept must be between $10k and $100M (prevents negatives)
    lower_bounds = np.ones(6) * 10000.0
    upper_bounds = np.ones(6) * 1.0e8

    # Run the Optimizer
    # method='trf' is robust against large steps and handles bounds
    res = least_squares(
        objective_function, 
        x0, 
        bounds=(lower_bounds, upper_bounds),
        diff_step=0.05,  # Finite difference step size (5%)
        verbose=2,       # Prints progress from scipy
        ftol=1e-3,       # Stop when cost change is small
        xtol=1e-3        # Stop when x change is small
    )

    print("\n✅ CALIBRATION COMPLETE")
    print("Final Intercepts:", np.round(res.x).astype(int))
    print("Final Counts:    ", np.round(res.fun + TARGET_VEC).astype(int))