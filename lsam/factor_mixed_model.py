import argparse
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

FACTORS = {"irregular": "irregular", "multiple": "multiple_lesions",
           "low_contrast": "low_contrast", "partial_edge": "partial_edge"}
META_COLS = ["case_id", "class", "variations", "noise_sigma", "blur_sigma", "intensity_scale"]


def read_any(path):
    df = pd.read_csv(path, sep=None, engine="python")   # detects comma or tab automatically
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--name", default="model")
    a = ap.parse_args()

    df = read_any(a.results)
    if a.metadata:
        meta = read_any(a.metadata)[["run_id", "file"] + META_COLS]
        df = df.drop(columns=[c for c in META_COLS if c in df.columns]).merge(meta, on=["run_id", "file"], how="inner")
    v = df["variations"].fillna("")
    for f, key in FACTORS.items():
        df[f] = v.str.contains(key).astype(int)

    vol = df.groupby(["run_id", "case_id"]).agg(
        HD95=("HD95", "mean"), DSC=("DSC", "mean"), ASD=("ASD", "mean"), cls=("class", "first"),
        noise=("noise_sigma", "first"), blur=("blur_sigma", "first"), scale=("intensity_scale", "first"),
        **{f: (f, "first") for f in FACTORS}).reset_index()
    vol["log_hd95"] = np.log1p(vol["HD95"])

    form = ("log_hd95 ~ irregular + multiple + low_contrast + partial_edge"
            " + multiple:irregular + multiple:low_contrast + multiple:partial_edge"
            " + noise + blur + scale + C(cls, Treatment('circle'))")
    m = smf.mixedlm(form, vol, groups=vol["run_id"]).fit(reml=True)

    terms = [t for t in m.params.index if t not in ("Intercept", "Group Var")]
    ci = m.conf_int()
    out = pd.DataFrame({
        "effect_x": np.exp(m.params[terms]),
        "ci_low": np.exp(ci.loc[terms, 0]),
        "ci_high": np.exp(ci.loc[terms, 1]),
        "p": m.pvalues[terms],
        "p_holm": multipletests(m.pvalues[terms], method="holm")[1],
    }).round(4)

    print(f"\n=== {a.name}: volumes = {len(vol)}, realizations = {vol.run_id.nunique()} ===")
    print(out.to_string())
    mult = vol[vol.multiple == 1]
    print("\nShare of multiple-lesion volumes with >=1 other factor:",
          round((mult[["irregular", "low_contrast", "partial_edge"]].sum(1) > 0).mean(), 3))
    for f in FACTORS:
        only = vol[(vol[f] == 1) & (vol[[g for g in FACTORS if g != f]].sum(1) == 0)]
        print(f"{f:13s} only: n={len(only):4d}  HD95={only.HD95.mean():.3f}  DSC={only.DSC.mean():.4f}")
    std = vol[vol[list(FACTORS)].sum(1) == 0]
    print(f"{'standard':13s}     : n={len(std):4d}  HD95={std.HD95.mean():.3f}  DSC={std.DSC.mean():.4f}")
    out.to_csv(f"mixed_model_{a.name}.csv")
    print(f"\nSaved mixed_model_{a.name}.csv")


if __name__ == "__main__":
    main()
