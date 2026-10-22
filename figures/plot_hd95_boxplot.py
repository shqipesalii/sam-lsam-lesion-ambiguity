"""Fig. 7 without significance stars: HD95 per slice by lesion class,
zero-shot SAM vs nnU-Net 2D on the ten V2 realizations (same slices)."""
import glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results_revision"
META = "/home/shqipe/Desktop/sam_phantom_project/SAM_10_runs/sam_10_runs_all_results.csv"
fix = lambda s: s.astype(str).str.replace(r"^(\d+)$", r"run\1", regex=True)

meta = pd.read_csv(META, sep=None, engine="python")[["run_id", "file", "class"]]
meta["run_id"] = fix(meta.run_id)

sam = pd.read_csv(f"{R}/sam_zeroshot_pad0_per_slice.csv"); sam["run_id"] = fix(sam.run_id)
sam = sam.merge(meta, on=["run_id", "file"])
nnf = sorted(glob.glob(f"{R}/nnunet2d*per_slice*.csv"))[0]
nn = pd.read_csv(nnf); nn["run_id"] = fix(nn.run_id)
nn = nn.merge(meta, on=["run_id", "file"])
print("SAM slices:", len(sam), "| nnU-Net 2D slices:", len(nn), "| nnU-Net file:", nnf)

classes = ["circle", "square", "triangle"]
for c in classes:
    print(f"{c:9s} SAM median {sam[sam['class']==c].HD95.median():.2f}  nnU-Net 2D median {nn[nn['class']==c].HD95.median():.2f}")

fig, ax = plt.subplots(figsize=(6, 3.6), dpi=300)
pos = np.arange(len(classes)); w = 0.35
for k, (d, lab, col) in enumerate(((sam, "SAM (zero-shot)", "#4C9BE8"), (nn, "nnU-Net 2D", "#E8833A"))):
    data = [d[d["class"] == c].HD95.values for c in classes]
    bp = ax.boxplot(data, positions=pos + (k - 0.5) * w, widths=w * 0.9, patch_artist=True,
                    flierprops=dict(marker=".", markersize=2, alpha=0.4), medianprops=dict(color="black"))
    for b in bp["boxes"]: b.set_facecolor(col); b.set_alpha(0.8)
    ax.plot([], [], color=col, lw=8, label=lab)
ax.set_xticks(pos); ax.set_xticklabels([c.capitalize() for c in classes])
ax.set_ylabel("HD95 (px)"); ax.set_yscale("symlog", linthresh=1)
ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(f"{R}/hd95_boxplot.png"); plt.savefig(f"{R}/hd95_boxplot.pdf")
print("saved", f"{R}/hd95_boxplot.png")
