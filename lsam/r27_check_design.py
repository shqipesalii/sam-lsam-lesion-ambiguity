"""Generation-only check of the revised structure-appearance design (no SAM needed, runs on CPU).

Regenerates the factorial cells with the original and the revised switch definitions and confirms that
  - cell A is identical in both designs (so it does not need to be rerun),
  - revised B and C differ from the original only in lesion contrast (masks, classes and nuisance parameters equal),
  - low contrast is absent from revised B and present at about 45% in revised C.
Usage: python r27_check_design.py [--runs 10] [--vols 150]
"""
import argparse
import numpy as np
from r27_factorial import create_base_phantom, generate_case

CELLS = {"A": (False, "v1", 7000), "B": (True, "v1", 7100), "C": (False, "v2", 7200)}


def gen(cell, design, run, nvol, base):
    structure, app, s0 = CELLS[cell]
    np.random.seed(s0 + run)
    aux = np.random.RandomState(100000 + s0 + run)
    return [generate_case(base, i <= 30, structure, app, design, aux) for i in range(1, nvol + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--vols", type=int, default=150)
    a = ap.parse_args()
    base = create_base_phantom()
    for cell in CELLS:
        img_eq = lbl_eq = meta_eq = 0
        low = {"original": 0, "revised": 0}
        n = 0
        for r in range(1, a.runs + 1):
            o, v = gen(cell, "original", r, a.vols, base), gen(cell, "revised", r, a.vols, base)
            for (io, lo, mo), (iv, lv, mv) in zip(o[30:], v[30:]):
                n += 1
                img_eq += np.array_equal(io, iv)
                lbl_eq += np.array_equal(lo, lv)
                meta_eq += all(mo[k] == mv[k] for k in ("cls", "noise", "blur", "scale"))
                low["original"] += "low_contrast" in mo["variations"]
                low["revised"] += "low_contrast" in mv["variations"]
        print(f"cell {cell}: {n} lesion volumes | identical images {img_eq}/{n}, masks {lbl_eq}/{n}, "
              f"class+nuisance {meta_eq}/{n} | low contrast original {low['original'] / n:.2f}, "
              f"revised {low['revised'] / n:.2f}")


if __name__ == "__main__":
    main()
