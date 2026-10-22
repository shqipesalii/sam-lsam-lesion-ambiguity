import glob
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

R = "results_revision"
META = "/home/shqipe/Desktop/sam_phantom_project/SAM_10_runs/sam_10_runs_all_results.csv"
M = ["DSC", "HD", "HD95", "ASD"]
rng = np.random.default_rng(0)
pd.set_option("display.width", 220)

meta = pd.read_csv(META, sep=None, engine="python")[["run_id", "file", "variations"]]
meta["multi"] = meta.variations.fillna("").str.contains("multiple_lesions")

pr = {}
for m in ("sam_zeroshot", "decoder_only", "lsam_full"):
    d = pd.read_csv(f"{R}/{m}_pad0_per_slice.csv").merge(meta, on=["run_id", "file"])
    for sub, mask in (("all", np.ones(len(d), bool)), ("multi", d.multi.values), ("single", ~d.multi.values)):
        pr[(m, sub)] = d[mask].groupby("run_id")[M].mean().sort_index()

nn_slice = glob.glob(f"{R}/nnunet2d*per_slice*.csv")
if nn_slice:
    d = pd.read_csv(nn_slice[0])
    d["run_id"] = d["run_id"].astype(str).str.replace(r"^(\d+)$", r"run\1", regex=True)
    d = d.merge(meta, on=["run_id", "file"], how="left")
    for sub, mask in (("all", np.ones(len(d), bool)), ("multi", d.multi.fillna(False).values),
                      ("single", ~d.multi.fillna(False).values)):
        pr[("nnunet2d", sub)] = d[mask].groupby("run_id")[M].mean().sort_index()
    print("nnU-Net 2D source:", nn_slice[0])
else:
    d = pd.read_csv(f"{R}/nnunet2d_medpy_per_run.csv")
    rc = [c for c in d.columns if c.lower() in ("run_id", "run")][0]
    d["run_id"] = d[rc].astype(str).str.replace(r"^(\d+)$", r"run\1", regex=True)
    pr[("nnunet2d", "all")] = d.set_index("run_id")[M].sort_index()
    print("nnU-Net 2D source: nnunet2d_medpy_per_run.csv (all slices only)")

print("\n=== Mean ± SD over 10 realizations ===")
for sub in ("all", "multi", "single"):
    print(f"\n-- {sub} slices --")
    for m in ("sam_zeroshot", "decoder_only", "lsam_full", "nnunet2d"):
        if (m, sub) in pr:
            r = pr[(m, sub)]
            print(f"{m:13s} " + "  ".join(f"{k} {r[k].mean():.4f} ± {r[k].std(ddof=1):.4f}" for k in M))

COMP = [("sam_zeroshot", "nnunet2d"), ("decoder_only", "sam_zeroshot"), ("lsam_full", "sam_zeroshot"),
        ("lsam_full", "decoder_only"), ("lsam_full", "nnunet2d"), ("decoder_only", "nnunet2d")]
rows = []
for sub in ("all", "multi", "single"):
    for k in M:
        fam = []
        for a, b in COMP:
            if (a, sub) not in pr or (b, sub) not in pr:
                continue
            x, y = pr[(a, sub)][k], pr[(b, sub)][k]
            idx = x.index.intersection(y.index); x, y = x.loc[idx], y.loc[idx]
            dd = (x - y).values
            lo, hi = np.percentile([rng.choice(dd, len(dd)).mean() for _ in range(10000)], [2.5, 97.5])
            better = (dd > 0).sum() if k == "DSC" else (dd < 0).sum()
            fam.append(dict(subset=sub, metric=k, comparison=f"{a} - {b}", n=len(dd), diff=dd.mean(),
                            ci_low=lo, ci_high=hi, runs_A_better=int(better),
                            p_wilcoxon=wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0,
                            p_ttest=ttest_rel(x.values, y.values).pvalue))
        if fam:
            for col in ("p_wilcoxon", "p_ttest"):
                for f, p in zip(fam, multipletests([f[col] for f in fam], method="holm")[1]):
                    f[col + "_holm"] = p
            rows += fam
res = pd.DataFrame(rows)
num = ["diff", "ci_low", "ci_high", "p_wilcoxon", "p_ttest", "p_wilcoxon_holm", "p_ttest_holm"]
res[num] = res[num].astype(float).round(4)
res.to_csv(f"{R}/lsam_comparison_paired.csv", index=False)
print("\n=== Paired differences A - B (Holm across comparisons within metric and subset) ===")
for sub in ("all", "multi", "single"):
    print(f"\n-- {sub} slices --")
    print(res[res.subset == sub].drop(columns="subset").to_string(index=False))
