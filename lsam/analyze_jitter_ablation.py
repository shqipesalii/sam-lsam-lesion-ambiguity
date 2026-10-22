import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

df = pd.read_csv("results_revision/box_jitter_ablation_per_slice.csv")
rng = np.random.default_rng(0)
pd.set_option("display.width", 200)
pr = (df.groupby(["model", "level", "run_id", "file"])[["DSC", "HD95", "ASD"]].mean()
        .groupby(["model", "level", "run_id"]).mean())
s = pr.groupby(["model", "level"]).agg(["mean", "std"]).round(4)
print("=== Mean (SD) over realizations ===\n", s.to_string())
rows = []
for level in sorted(df.level.unique()):
    for k in ("HD95", "ASD"):
        fam = []
        ref = pr.loc[("ablation_full", level)][k]
        for v in ["no_boundary_loss", "no_low_contrast", "no_boundary_block", "no_multi_context", "simple_adapter"]:
            d = (pr.loc[(v, level)][k] - ref).values
            lo, hi = np.percentile([rng.choice(d, 10).mean() for _ in range(10000)], [2.5, 97.5])
            fam.append(dict(level=level, metric=k, variant=v, diff=d.mean(), ci_low=lo, ci_high=hi,
                            runs_worse=int((d > 0).sum()), p=wilcoxon(d).pvalue))
        for f, p in zip(fam, multipletests([f["p"] for f in fam], method="holm")[1]):
            f["p_holm"] = p
        rows += fam
res = pd.DataFrame(rows).round(4)
res.to_csv("results_revision/box_jitter_ablation_paired.csv", index=False)
print("\n=== Variant minus matched full (Holm across 5 variants) ===\n", res.to_string(index=False))
