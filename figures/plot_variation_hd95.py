"""Fig.: HD95 of zero-shot SAM by active variation on V2.
Each box: the 10 per-realization means (same aggregation as variations_with_stdv.py); diamond = mean.
Input: SAM_10_runs/sam_10_runs_all_results.csv. Output: variation_hd95.png / .pdf"""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("SAM_10_runs/sam_10_runs_all_results.csv")
groups = [  # (key in 'variations', label, colour)
    ("standard",         "Standard",                 "#5B7A99"),
    ("strong_noise",     "Strong\nnoise",            "#F5C26B"),
    ("low_contrast",     "Low\ncontrast",            "#F0A05F"),
    ("irregular",        "Irregular\nboundaries",    "#EB7A57"),
    ("partial_edge",     "Peripheral\ndisplacement", "#DC4E41"),
    ("multiple_lesions", "Multiple\nlesions",        "#C0344A"),
]
data = []
for key, _, _ in groups:
    sub = df[df.variations == "standard"] if key == "standard" else df[df.variations.str.contains(key, na=False)]
    data.append(sub.groupby("run_id").HD95.mean().values)

fig, ax = plt.subplots(figsize=(7.2, 3.3))
bp = ax.boxplot(data, widths=0.5, patch_artist=True, whis=(0, 100), showfliers=False, showmeans=True,
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(color="black", linewidth=1.3), capprops=dict(color="black", linewidth=1.3),
                boxprops=dict(edgecolor="black", linewidth=1.3),
                meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=7))
for patch, (_, _, c) in zip(bp["boxes"], groups):
    patch.set_facecolor(c); patch.set_alpha(0.9)
ax.set_xticks(range(1, len(groups) + 1))
ax.set_xticklabels([g[1] for g in groups])
ax.set_ylabel("HD95")
ax.set_title("SAM variation-level robustness analysis", fontweight="bold")
ax.yaxis.grid(True, color="#dddddd", linewidth=0.8); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("variation_hd95.png", dpi=300); plt.savefig("variation_hd95.pdf")
for (_, lab, _), d in zip(groups, data):
    print(f"{lab.replace(chr(10), ' '):25s} n_runs={len(d):2d}  mean HD95 {d.mean():.2f}")
print("Saved variation_hd95.png / .pdf")
