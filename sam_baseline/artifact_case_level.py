import os, re
import numpy as np, pandas as pd, cv2
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from medpy.metric.binary import dc, hd, hd95, assd

PRED = "Artifact_SAM_Predictions"
DATA = "Artifact_SAM_Datasets"
EMPTY = float(np.hypot(256, 256))
M = ["DSC", "HD", "HD95", "ASD"]
rng = np.random.default_rng(0)

def metrics(p, g):
    if p.sum() == 0:
        return 0.0, EMPTY, EMPTY, EMPTY
    return dc(p, g), hd(p, g), hd95(p, g), assd(p, g)

def gt_dir(cond):
    for d in (f"{DATA}/{cond}/masks", f"{DATA}/V2_moderate_run01/masks"):
        if os.path.isdir(d):
            return d
    raise FileNotFoundError("no GT masks found")

conds = ["V2_original"] + sorted(d for d in os.listdir(PRED) if re.match(r"V2_(moderate|severe)_run\d+$", d))
rows = []
for c in conds:
    gdir = gt_dir(c)
    for f in sorted(os.listdir(f"{PRED}/{c}")):
        if not f.endswith(".png"):
            continue
        g = (cv2.imread(os.path.join(gdir, f), 0) > 0).astype(np.uint8)
        p = (cv2.imread(os.path.join(PRED, c, f), 0) > 0).astype(np.uint8)
        sev = "original" if c == "V2_original" else c.split("_")[1]
        r = dict(condition=c, severity=sev, file=f, case=f.split("_")[0], lesion=bool(g.sum()))
        if g.sum():
            r.update(zip(M, metrics(p, g)))
        else:
            r["FP"] = int(p.sum() > 0)
        rows.append(r)
    print(c, "done")
df = pd.DataFrame(rows)
df.to_csv("artifact_case_level_per_slice.csv", index=False)

les = df[df.lesion]
print("\nlesion slices per condition:", les.groupby("condition").size().unique(),
      "| lesion-free slices:", df[~df.lesion].groupby("condition").size().unique())
print("false-positive rate on lesion-free slices:\n", df[~df.lesion].groupby("severity").FP.mean().round(4).to_string())

sl = les.groupby(["severity", "case", "file"])[M].mean()
vol = sl.groupby(["severity", "case"]).mean()
print(f"\nvolumes: {vol.loc['original'].shape[0]}")

out = []
for s in ("moderate", "severe"):
    d_all = vol.loc[s] - vol.loc["original"]
    for k in M:
        d = d_all[k].dropna().values
        lo, hi = np.percentile([rng.choice(d, len(d)).mean() for _ in range(10000)], [2.5, 97.5])
        worse = (d < 0).mean() if k == "DSC" else (d > 0).mean()
        out.append(dict(severity=s, metric=k, original=vol.loc["original"][k].mean(),
                        perturbed=vol.loc[s][k].mean(), change=d.mean(), ci_low=lo, ci_high=hi,
                        dz=d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan,
                        share_volumes_worse=worse, p=wilcoxon(d).pvalue if np.any(d != 0) else 1.0))
res = pd.DataFrame(out)
res["p_holm"] = multipletests(res.p, method="holm")[1]
res = res.round(4)
res.to_csv("artifact_case_level_summary.csv", index=False)
pd.set_option("display.width", 200)
print("\n=== Paired case-level change from undegraded (volumes; Holm over 2 severities x 4 metrics) ===")
print(res.to_string(index=False))

runm = les[les.severity != "original"].groupby(["severity", "condition"])[M].mean()
print("\n=== Run-to-run variability of the perturbation procedure (SD of run means) ===")
print(runm.groupby("severity").agg(["mean", "std"]).round(4).to_string())
