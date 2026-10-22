"""Forest plot of adjusted factor effects (replaces Fig. 8). Input: mixed_model_SAM.csv from factor_mixed_model.py"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "mixed_model_SAM.csv"
df = pd.read_csv(src, index_col=0)

rows = [  # (label, term, group)
    ("Multiple lesions", "multiple", "Ambiguity factors"),
    ("Partial-edge visibility", "partial_edge", "Ambiguity factors"),
    ("Irregular boundaries", "irregular", "Ambiguity factors"),
    ("Low contrast", "low_contrast", "Ambiguity factors"),
    ("Multiple × partial edge", "multiple:partial_edge", "Interactions"),
    ("Multiple × low contrast", "multiple:low_contrast", "Interactions"),
    ("Multiple × irregular", "multiple:irregular", "Interactions"),
    ("Blur σ (per unit)", "blur", "Nuisance parameters"),
    ("Noise σ (per unit)", "noise", "Nuisance parameters"),
    ("Intensity scaling", "scale", "Nuisance parameters"),
    ("Square vs. circle", "C(cls, Treatment('circle'))[T.square]", "Lesion class"),
    ("Triangle vs. circle", "C(cls, Treatment('circle'))[T.triangle]", "Lesion class"),
]

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(3.5, 3.6))
y, ypos, labels, group_y = 0, [], [], []
last_group = None
for label, term, group in rows:
    if group != last_group:
        if last_group is not None:
            y += 0.6
        group_y.append((y, group))
        y += 1
        last_group = group
    ypos.append(y); labels.append(label); y += 1
ypos = np.array(ypos)

for (label, term, _), yy in zip(rows, ypos):
    r = df.loc[term]
    sig = r["p_holm"] < 0.05
    color = "#1f4e9c" if sig else "#8a8a8a"
    ax.plot([r["ci_low"], r["ci_high"]], [yy, yy], color=color, lw=1.2, solid_capstyle="butt")
    ax.plot(r["effect_x"], yy, "o", ms=4.2, color=color, mec="white", mew=0.5, zorder=3)
    txt = f"×{r['effect_x']:.2f}"
    ax.text(9.5, yy, txt, va="center", ha="left", fontsize=7, color=color,
            fontweight="bold" if sig else "normal")

for gy, g in group_y:
    ax.text(-0.03, gy, g, fontsize=7.5, fontweight="bold", va="center", ha="right",
            transform=ax.get_yaxis_transform())

ax.axvline(1, color="black", lw=0.6, ls="--", zorder=0)
ax.set_xscale("log")
ax.set_xlim(0.45, 9.5)
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4, 8]))
ax.xaxis.set_major_formatter(FixedFormatter(["0.5", "1", "2", "4", "8"]))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_yticks(ypos); ax.set_yticklabels(labels)
ax.set_ylim(y, -0.8)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("Multiplicative effect on HD95+1 (95% CI, log scale)")
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("Fig_factor_effects.pdf", bbox_inches="tight")
fig.savefig("Fig_factor_effects.png", dpi=600, bbox_inches="tight")
print("Saved Fig_factor_effects.pdf / .png")
