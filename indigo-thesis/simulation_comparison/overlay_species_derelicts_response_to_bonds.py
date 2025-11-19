#!/usr/bin/env python3
"""
Overlay comparison: sns_test_2 vs intensive for species_and_derelicts_response_to_bonds
Reads raw species_data_*.json files from the Results folders and re-plots
both datasets on the SAME axes.

- Top row: S, Su, Sns
- Bottom row: N_521kg, N_700kg, N_20kg (if present)
- Baseline (no bond pattern) plotted at $0k
- Styling:
  - Color by dataset: sns_test_2 = C0 (blue), intensive = C1 (orange)
  - Linestyle by lifetime: 5yr = solid, 25yr = dashed

Output: indigo-thesis/simulation_comparison/overlay_species_and_derelicts_response_to_bonds.png
"""

import os
import re
import json
from typing import Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt

BASE = "/Users/indigobrownhall/Code/OPUS"
SETS = {
    "sns_test_2": os.path.join(BASE, "Results", "sns_test_2"),
    "intensive": os.path.join(BASE, "Results", "intensive"),
}
OUT_PATH = os.path.join(BASE, "indigo-thesis", "simulation_comparison", "overlay_species_and_derelicts_response_to_bonds.png")

SERIES_KEYS = ["S", "Su", "Sns", "N_521kg", "N_700kg", "N_20kg"]


def sum_final_year(values_by_year: Dict[str, List[float]]) -> float:
    """Sum across shells at the FINAL available year."""
    if not values_by_year:
        return 0.0
    year_keys = sorted(map(int, values_by_year.keys()))
    final_year = str(year_keys[-1])
    arr = np.array(values_by_year[final_year], dtype=float)
    return float(np.sum(arr))


def harvest_series(root: str) -> Dict[str, Dict[int, Dict[str, List[Tuple[int, float]]]]]:
    """
    Walk a Results root, read species_data_*.json for each scenario, extract final-year totals
    for SERIES_KEYS. Returns nested dict:
    { series_key: { lifetime: { set_label: [(bond_k, total), ...] } } }
    Here set_label is implicit in the caller; this returns {series_key: {lifetime: [(bond_k, total), ...]}}
    """
    out: Dict[str, Dict[int, List[Tuple[int, float]]]] = {k: {5: [], 25: []} for k in SERIES_KEYS}

    for entry in os.listdir(root):
        scen_dir = os.path.join(root, entry)
        if not os.path.isdir(scen_dir):
            continue
        # find species_data_*.json in this directory
        json_file = None
        for fname in os.listdir(scen_dir):
            if fname.startswith("species_data_") and fname.endswith(".json"):
                json_file = os.path.join(scen_dir, fname)
                break
        if not json_file:
            continue

        # parse bond/lifetime from directory name (preferred) or filename
        m = re.search(r"bond_(\d+)k_(\d+)yr", entry.lower())
        if not m:
            # baseline $0k, default 5yr unless specified in name
            bond_k = 0
            m2 = re.search(r"(5|25)yr", entry.lower())
            lifetime = int(m2.group(1)) if m2 else 5
        else:
            bond_k = int(m.group(1))
            lifetime = int(m.group(2))

        with open(json_file, "r") as f:
            data = json.load(f)

        for key in SERIES_KEYS:
            if key in data and isinstance(data[key], dict):
                total = sum_final_year(data[key])
                out[key][lifetime].append((bond_k, total))

    # sort by bond
    for key in SERIES_KEYS:
        for life in (5, 25):
            out[key][life].sort(key=lambda x: x[0])
    return out


def main():
    results = {}
    for label, root in SETS.items():
        results[label] = harvest_series(root)

    # plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    # No subplot titles per request; we keep ordering only
    panels = [
        (0, 0, "S"),
        (0, 1, "Su"),
        (0, 2, "Sns"),
        (1, 0, "N_521kg"),
        (1, 1, "N_700kg"),
        (1, 2, "N_20kg"),
    ]

    # Rename datasets on plots (if later legends/titles are added)
    dataset_names = {"sns_test_2": "Uncontrolled", "intensive": "Controlled"}
    colors = {"sns_test_2": "C0", "intensive": "C1"}
    linestyles = {5: "-", 25: "--"}

    for r, c, key in panels:
        ax = axes[r][c]
        any_plotted = False
        for dataset_key in SETS.keys():
            series = results[dataset_key].get(key, {5: [], 25: []})
            for life in (5, 25):
                pts = series.get(life, [])
                if not pts:
                    continue
                any_plotted = True
                xs = [b for b, _ in pts]
                ys = [y for _, y in pts]
                ax.plot(xs, ys, color=colors[dataset_key], linestyle=linestyles[life], marker='o')
        ax.set_xlabel('Bond Amount ($k)')

        # Axis labeling adjustments:
        if r == 0:
            # top row: remove the word "Species"
            ax.set_ylabel(f'{key} Count')
        else:
            # bottom row: map to S/Su Derelicts where appropriate
            if key == 'N_521kg':
                ax.set_ylabel('S Derelicts Count')
            elif key == 'N_700kg':
                ax.set_ylabel('Su Derelicts Count')
            else:
                ax.set_ylabel('Derelicts Count')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)


        # remove legends and in-plot labels per request
        # (keep axes only)
        # if not any_plotted:
        #     ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=300)
    print(f"Saved overlay comparison figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
