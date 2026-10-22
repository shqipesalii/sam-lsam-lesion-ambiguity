import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

R = "results_revision"
META = "/home/shqipe/Desktop/sam_phantom_project/SAM_10_runs/sam_10_runs_all_results.csv"
MODELS = ["sam_zeroshot", "decoder_only", "lsam_full", "ablation_full", "no_boundary_loss",
          "no_low_contrast", "no_boundary_block", "no_multi_context", "simple_adapter"]
VARIANTS = ["no_boundary_loss", "no_low_contrast", "no_boundary_block", "no_multi_context", "simple_adapter"]
METRICS = ["DSC", "HD", "HD95", "ASD"]
rng = np.random.default_rng(0)

meta = pd.read_csv(META, sep=None, engine="python")[["run_id", "file", "variations"]]
meta["multi"] = meta["variations"].fillna("").str.contains("multiple_lesions")

per_run = {}
for m in MODELS:
    d = pd.read_csv(f"{R}/{m}_pad0_per_slice.csv").merge(meta, on=["run_id", "file"])
    for sub, mask in (("all", np.ones(len(d), bool)), ("multi", d.multi.values), ("single", ~d.multi.values)):
        per_run[(m, sub)] = d[mask].groupby("run_id")[METRICS].mean().sort_index()

rows = []
for m in MODELS:
    for sub in ("all", "multi", "single"):
        r = per_run[(m, sub)]
        rows.append({"model": m, "subset": sub,
                     **{k: f"{r[k].mean():.4f} ± {r[k].std(ddof=1):.4f}" for k in METRICS}})
tab = pd.DataFrame(rows)
tab.to_csv(f"{R}/ablation_table.csv", index=False)
print("=== Mean ± SD over 10 realizations ===")
for sub in ("all", "multi", "single"):
    print(f"\n-- {sub} slices --")
    print(tab[tab.subset == sub].drop(columns="subset").to_string(index=False))

def paired(a, b):
    d = a.values - b.values
    boot = [rng.choice(d, len(d)).mean() for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    pw = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return d.mean(), lo, hi, pw, ttest_rel(a.values, b.values).pvalue, d

out = []
for sub in ("all", "multi", "single"):
    for k in METRICS:
        fam = []
        for v in VARIANTS:
            mean, lo, hi, pw, pt, d = paired(per_run[(v, sub)][k], per_run[("ablation_full", sub)][k])
            worse = (d < 0).sum() if k == "DSC" else (d > 0).sum()
            fam.append(dict(comparison=f"{v} - ablation_full", subset=sub, metric=k, diff=mean,
                            ci_low=lo, ci_high=hi, runs_worse=int(worse), p_wilcoxon=pw, p_ttest=pt))
        pw_h = multipletests([f["p_wilcoxon"] for f in fam], method="holm")[1]
        pt_h = multipletests([f["p_ttest"] for f in fam], method="holm")[1]
        for f, a_, b_ in zip(fam, pw_h, pt_h):
            f["p_wilcoxon_holm"], f["p_ttest_holm"] = a_, b_
        out += fam
        mean, lo, hi, pw, pt, d = paired(per_run[("lsam_full", sub)][k], per_run[("decoder_only", sub)][k])
        worse = (d < 0).sum() if k == "DSC" else (d > 0).sum()
        out.append(dict(comparison="lsam_full - decoder_only", subset=sub, metric=k, diff=mean, ci_low=lo,
                        ci_high=hi, runs_worse=int(worse), p_wilcoxon=pw, p_ttest=pt,
                        p_wilcoxon_holm=np.nan, p_ttest_holm=np.nan))
res = pd.DataFrame(out)
num = ["diff", "ci_low", "ci_high", "p_wilcoxon", "p_ttest", "p_wilcoxon_holm", "p_ttest_holm"]
res[num] = res[num].astype(float).round(4)
res.to_csv(f"{R}/ablation_paired_tests.csv", index=False)
pd.set_option("display.width", 200)
print("\n=== Paired differences (variant minus matched full; n = 10, Holm across 5 variants) ===")
for sub in ("all", "multi", "single"):
    print(f"\n-- {sub} slices --")
    print(res[(res.subset == sub) & res.metric.isin(["HD95", "ASD", "DSC"])]
          .drop(columns="subset").to_string(index=False))
print("\nSaved:", f"{R}/ablation_table.csv,", f"{R}/ablation_paired_tests.csv")
