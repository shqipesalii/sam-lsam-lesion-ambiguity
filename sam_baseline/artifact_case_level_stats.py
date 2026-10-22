import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
M = ["DSC", "HD", "HD95", "ASD"]
rng = np.random.default_rng(0)
les = pd.read_csv("artifact_case_level_per_slice.csv")
les = les[les.lesion]
print("slices per condition:", les.groupby("condition").size().unique(), "| conditions:", les.condition.nunique())
print("empty predictions per severity:\n", (les.DSC == 0).groupby(les.severity).sum().to_string())
vol = les.groupby(["severity", "case", "file"])[M].mean().groupby(["severity", "case"]).mean()
print("volumes:", vol.loc["original"].shape[0])
out = []
for s in ("moderate", "severe"):
    d_all = vol.loc[s] - vol.loc["original"]
    for k in M:
        d = d_all[k].values
        lo, hi = np.percentile([rng.choice(d, len(d)).mean() for _ in range(10000)], [2.5, 97.5])
        worse = (d < 0).mean() if k == "DSC" else (d > 0).mean()
        out.append(dict(severity=s, metric=k, original=vol.loc["original"][k].mean(), perturbed=vol.loc[s][k].mean(),
                        change=d.mean(), ci_low=lo, ci_high=hi, dz=d.mean() / d.std(ddof=1),
                        share_volumes_worse=worse, p=wilcoxon(d).pvalue))
res = pd.DataFrame(out); res["p_holm"] = multipletests(res.p, method="holm")[1]
res = res.round(4); res.to_csv("artifact_case_level_summary.csv", index=False)
pd.set_option("display.width", 200)
print("\n=== Paired case-level change from undegraded (volumes; Holm over 2 severities x 4 metrics) ===")
print(res.to_string(index=False))
print("\n=== Undegraded, case-level mean ± SD over volumes ===")
print(vol.loc["original"].agg(["mean", "std"]).round(4).to_string())
runm = les[les.severity != "original"].groupby(["severity", "condition"])[M].mean()
print("\n=== Run-to-run variability of the perturbation procedure (mean and SD of the 10 run means) ===")
print(runm.groupby("severity").agg(["mean", "std"]).round(4).to_string())
