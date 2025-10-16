import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from typing import Dict, Any

# ------------------------------------------------------------
# Fast cached loader: convert .mat -> .npz once, then read .npz (much faster)
# ------------------------------------------------------------

def load_lookup_cached(mat_path: str, cache_npz_path: str = None) -> Dict[str, Any]:
    """
    Load lookup struct from a MATLAB .mat file, but cache as .npz for fast reloads.
    If cache exists and is newer than the .mat, load the cache. Otherwise convert.
    Returns a dict with plain NumPy arrays.
    """
    if cache_npz_path is None:
        base, _ = os.path.splitext(mat_path)
        cache_npz_path = base + ".npz"

    use_cache = False
    if os.path.exists(cache_npz_path) and os.path.exists(mat_path):
        use_cache = os.path.getmtime(cache_npz_path) >= os.path.getmtime(mat_path)

    if use_cache:
        npz = np.load(cache_npz_path)
        return {k: npz[k] for k in npz.files}

    # Convert from .mat -> .npz
    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    lookup = data["lookup"]

    out = {
        "years":            np.array(lookup.years, dtype=int).flatten(),
        "apogee_alts_km":   np.array(lookup.apogee_alts_km).flatten(),
        "perigee_alts_km":  np.array(lookup.perigee_alts_km).flatten(),
        "coef_logquad":     np.array(lookup.coef_logquad),  # (ny, na, 3)
        "R2_log":           np.array(lookup.R2_log),        # (ny, na)
        "lifetimes_years":  np.array(lookup.lifetimes_years),
        "decay_alt_km":     np.array(lookup.decay_alt_km).item() if np.size(lookup.decay_alt_km)==1 else np.array(lookup.decay_alt_km)
    }

    # Save uncompressed for fastest reload (or use savez_compressed for smaller files)
    np.savez(cache_npz_path, **out)
    return out

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def inv_logquad_for_y(p, y_target, xmin, xmax):
    """Solve for x given coefficients [a,b,c] of log(y)=a*x^2+b*x+c."""
    if p is None or np.any(np.isnan(p)):
        return np.nan
    a, b, c = p
    c = c - np.log(y_target)
    D = b**2 - 4*a*c
    if D < 0:
        return np.nan
    roots = np.array([(-b + np.sqrt(D)) / (2*a), (-b - np.sqrt(D)) / (2*a)])
    roots = roots[(roots >= xmin) & (roots <= xmax)]
    if roots.size == 0:
        return np.nan
    return roots[np.argmin(np.abs(roots - 0.5 * (xmin + xmax)))]


def required_perigee_curve_for_year(coef_logquad, R2_log, years, apogee_alts, perigee_alts, iy, target_years):
    """Return required perigee (km) vs apogee (km) for years[iy]."""
    perigee_min, perigee_max = float(perigee_alts.min()), float(perigee_alts.max())
    req_perigee = np.full_like(apogee_alts, np.nan, dtype=float)
    for ia, apogee in enumerate(apogee_alts):
        p = coef_logquad[iy, ia, :]
        if np.any(np.isnan(p)):
            continue
        hp = inv_logquad_for_y(p, target_years, perigee_min, perigee_max)
        # Skip invalid/physically impossible solutions
        if np.isnan(hp) or (hp > apogee) or (hp < perigee_min) or (hp > perigee_max):
            continue
        # Optional: reject low-quality fits
        if R2_log.size and not np.isnan(R2_log[iy, ia]) and R2_log[iy, ia] < 0.95:
            continue
        req_perigee[ia] = hp
    return req_perigee


# ------------------------------------------------------------
# Main: load data (with cache), then plot for all available years
# ------------------------------------------------------------

MAT_PATH = "disposal_lookup_24_26.mat"   # adjust path if needed
cache = load_lookup_cached(MAT_PATH)      # creates disposal_lookup_24_26.npz on first run

years           = cache["years"]
apogee_alts     = cache["apogee_alts_km"]
perigee_alts    = cache["perigee_alts_km"]
coef_logquad    = cache["coef_logquad"]
R2_log          = cache["R2_log"]
# lifetimes_years = cache["lifetimes_years"]  # available if needed

# Plot for all years
TARGET_LIFETIME = 5.0
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(years)))
handles = []
labels = []

for iy, year in enumerate(years):
    curve = required_perigee_curve_for_year(coef_logquad, R2_log, years, apogee_alts, perigee_alts, iy, TARGET_LIFETIME)
    if np.all(np.isnan(curve)):
        continue
    h, = plt.plot(apogee_alts, curve, "o-", lw=1.8, ms=4, color=colors[iy])
    handles.append(h)
    labels.append(f"{year}")

plt.xlabel("Apogee altitude (km)")
plt.ylabel("Required perigee altitude (km)")
plt.title(f"Required perigee for {TARGET_LIFETIME:.0f}-year disposal — all available years")
plt.grid(True, alpha=0.35)
if handles:
    plt.legend(handles, labels, title="Start year", fontsize=9, ncol=2)
plt.tight_layout()
plt.show()

# (Optional) Save the plot
# plt.savefig("required_perigee_all_years.png", dpi=150)