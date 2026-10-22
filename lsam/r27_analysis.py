import numpy as np, pandas as pd
import statsmodels.formula.api as smf
R = "results_revision"
M = ["DSC", "HD", "HD95", "ASD"]
rng = np.random.default_rng(0)
pd.set_option("display.width", 200)
new = pd.read_csv(f"{R}/r27_per_slice.csv")[["cell", "structure", "appearance", "run", "case_id", "prompt"] + M]
parts = []
for prompt, path in (("merged", f"{R}/sam_zeroshot_pad0_per_slice.csv"), ("per_lesion", f"{R}/sam_per_lesion_protocol_per_slice.csv")):
    d = pd.read_csv(path); d["case_id"] = d.file.str.split("_").str[0]
    d = d.assign(cell="D", structure=1, appearance="v2", run="D" + d.run_id.str.replace("run", ""), prompt=prompt)
    parts.append(d[["cell", "structure", "appearance", "run", "case_id", "prompt"] + M])
df = pd.concat([new] + parts, ignore_index=True)
names = {"A": "no structure, V1 appearance", "B": "structure, V1 appearance", "C": "no structure, V2 appearance", "D": "structure, V2 appearance (V2)"}
for prompt in ("merged", "per_lesion"):
    sub = df[df.prompt == prompt]; run = sub.groupby(["cell", "run"])[M].mean()
    print(f"\n=== {prompt}: mean ± SD over realizations ===")
    for c in "ABCD":
        r = run.loc[c]; print(f"{c} {names[c]:32s} n={len(r):2d}  " + "  ".join(f"{k} {r[k].mean():.4f} ± {r[k].std(ddof=1):.4f}" for k in M))
    print("differences of cell means (bootstrap 95% CI over realizations):")
    for a_, b_, lab in (("B", "A", "structure | V1 appearance"), ("D", "C", "structure | V2 appearance"), ("C", "A", "appearance | no structure"), ("D", "B", "appearance | structure")):
        for k in ("DSC", "HD95"):
            x, y = run.loc[a_][k].values, run.loc[b_][k].values
            lo, hi = np.percentile([rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(10000)], [2.5, 97.5])
            print(f"  {lab:28s} {k:4s} {x.mean() - y.mean():+8.4f} [{lo:+.4f}, {hi:+.4f}]")
    vol = sub.groupby(["cell", "structure", "appearance", "run", "case_id"]).HD95.mean().reset_index()
    vol["log_hd95"] = np.log1p(vol.HD95); vol["app_v2"] = (vol.appearance == "v2").astype(int)
    fit = smf.mixedlm("log_hd95 ~ structure * app_v2", vol, groups=vol.run).fit(reml=True)
    ci = fit.conf_int(); t = ["structure", "app_v2", "structure:app_v2"]
    out = pd.DataFrame({"effect_x": np.exp(fit.params[t]), "ci_low": np.exp(ci.loc[t, 0]), "ci_high": np.exp(ci.loc[t, 1]), "p": fit.pvalues[t]}).round(4)
    print(f"mixed model on log(HD95+1), volumes = {len(vol)} (reference = cell A):"); print(out.to_string())
