import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

P = "/home/shqipe/Desktop/sam_phantom_project"
PER_LESION = "results_revision/per_lesion_vs_merged_per_slice.csv"
META = f"{P}/SAM_10_runs/sam_10_runs_all_results.csv"
SAM_NEW = "results_revision/sam_zeroshot_pad0_per_slice.csv"
METRICS = ["DSC", "HD", "HD95", "ASD"]
rng = np.random.default_rng(0)


def boot_ci(d, n=10000):
    b = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


# 1. paired comparison on multiple-lesion slices (unit = realization, n = 10)
pl = pd.read_csv(PER_LESION)
run = pl.groupby(["run_id", "setting"])[METRICS].mean().unstack("setting")
print("=== 1. Multiple-lesion slices: merged vs per-lesion (per-run means, n = 10) ===")
print(f"slices: {pl[pl.setting == 'merged'].shape[0]}, "
      f"single-component slices: {int((pl[pl.setting == 'merged'].n_components == 1).sum())}")
rows, pt, pw = [], [], []
for m in METRICS:
    mg, pr = run[(m, "merged")].values, run[(m, "per_lesion")].values
    d = pr - mg
    lo, hi = boot_ci(d)
    pt.append(ttest_rel(pr, mg).pvalue)
    pw.append(wilcoxon(d).pvalue)
    rows.append(dict(metric=m, merged=f"{mg.mean():.4f} ± {mg.std(ddof=1):.4f}",
                     per_lesion=f"{pr.mean():.4f} ± {pr.std(ddof=1):.4f}",
                     diff=round(d.mean(), 4), ci_low=round(lo, 4), ci_high=round(hi, 4),
                     runs_improved=int((d > 0).sum() if m == "DSC" else (d < 0).sum())))
res = pd.DataFrame(rows)
res["p_ttest_holm"] = multipletests(pt, method="holm")[1]
res["p_wilcoxon_holm"] = multipletests(pw, method="holm")[1]
print(res.to_string(index=False))
res.to_csv("results_revision/R22_paired_comparison.csv", index=False)

# 2. SAM overall V2 with multiple-lesion slices prompted per lesion
meta = pd.read_csv(META, sep=None, engine="python")
use_new = os.path.exists(SAM_NEW)
base = pd.read_csv(SAM_NEW) if use_new else meta.copy()
base = base[["run_id", "file"] + METRICS]
per = pl[pl.setting == "per_lesion"][["run_id", "file"] + METRICS]
overall = base.set_index(["run_id", "file"])
overall.update(per.set_index(["run_id", "file"]))
overall = overall.reset_index()
cols = METRICS if use_new else ["DSC", "HD", "HD95"]
print("\n=== 2. SAM overall V2, multiple-lesion slices prompted per lesion ===")
print("source for other slices:", SAM_NEW if use_new else "original SAM results (ASD skipped)")
for name, df in (("merged (original protocol)", base), ("per-lesion prompting", overall)):
    r = df.groupby("run_id")[cols].mean().agg(["mean", "std"])
    print(f"  {name:28s} " + "  ".join(f"{m} {r.loc['mean', m]:.4f} ± {r.loc['std', m]:.4f}" for m in cols))
overall.to_csv("results_revision/sam_per_lesion_protocol_per_slice.csv", index=False)

# 3. mixed model under per-lesion prompting
F = {"irregular": "irregular", "multiple": "multiple_lesions",
     "low_contrast": "low_contrast", "partial_edge": "partial_edge"}
mm = overall.drop(columns=[c for c in ["DSC", "ASD", "HD"] if c in overall.columns]).merge(
    meta[["run_id", "file", "case_id", "class", "variations", "noise_sigma", "blur_sigma", "intensity_scale"]],
    on=["run_id", "file"])
v = mm["variations"].fillna("")
for f, k in F.items():
    mm[f] = v.str.contains(k).astype(int)
vol = mm.groupby(["run_id", "case_id"]).agg(HD95=("HD95", "mean"), cls=("class", "first"),
                                            noise=("noise_sigma", "first"), blur=("blur_sigma", "first"),
                                            scale=("intensity_scale", "first"),
                                            **{f: (f, "first") for f in F}).reset_index()
vol["log_hd95"] = np.log1p(vol["HD95"])
form = ("log_hd95 ~ irregular + multiple + low_contrast + partial_edge"
        " + multiple:irregular + multiple:low_contrast + multiple:partial_edge"
        " + noise + blur + scale + C(cls, Treatment('circle'))")
fit = smf.mixedlm(form, vol, groups=vol["run_id"]).fit(reml=True)
terms = [t for t in fit.params.index if t not in ("Intercept", "Group Var")]
ci = fit.conf_int()
out = pd.DataFrame({"effect_x": np.exp(fit.params[terms]), "ci_low": np.exp(ci.loc[terms, 0]),
                    "ci_high": np.exp(ci.loc[terms, 1]), "p": fit.pvalues[terms],
                    "p_holm": multipletests(fit.pvalues[terms], method="holm")[1]}).round(4)
print(f"\n=== 3. Mixed model, per-lesion prompting (volumes = {len(vol)}) ===")
print(out.to_string())
out.to_csv("mixed_model_SAM_per_lesion.csv")
print("\nSaved: results_revision/R22_paired_comparison.csv, "
      "results_revision/sam_per_lesion_protocol_per_slice.csv, mixed_model_SAM_per_lesion.csv")
