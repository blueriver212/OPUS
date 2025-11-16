#!/usr/bin/env python3
"""
Create side-by-side comparison of species_and_derelicts_response_to_bonds from
- Results/sns_test_2/comparisons/
- Results/intensive/comparisons/

Outputs a combined PNG in this folder.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.image import imread

BASE = "/Users/indigobrownhall/Code/OPUS"
A_PATH = os.path.join(BASE, "Results", "sns_test_2", "comparisons", "species_and_derelicts_response_to_bonds.png")
B_PATH = os.path.join(BASE, "Results", "intensive", "comparisons", "species_and_derelicts_response_to_bonds.png")
OUT_PATH = os.path.join(BASE, "indigo-thesis", "simulation_comparison", "species_and_derelicts_response_to_bonds_comparison.png")


def main():
    img_a = imread(A_PATH)
    img_b = imread(B_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    axes[0].imshow(img_a)
    axes[0].axis("off")
    axes[0].set_title("sns_test_2: Species and Derelicts Response to Bonds", fontsize=14, fontweight="bold")

    axes[1].imshow(img_b)
    axes[1].axis("off")
    axes[1].set_title("intensive: Species and Derelicts Response to Bonds", fontsize=14, fontweight="bold")

    fig.suptitle("Comparison: sns_test_2 vs intensive", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=200)
    print(f"Saved comparison to {OUT_PATH}")


if __name__ == "__main__":
    main()
