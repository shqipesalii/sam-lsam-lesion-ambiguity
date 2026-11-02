"""Structure-appearance factorial analysis.

--design original : cells A-C from r27_per_slice.csv (structure switch includes low contrast; first revision).
--design revised  : structure = irregular, multiple, peripheral displacement; appearance = low contrast + noise,
                    blur, intensity scaling. Cell A is identical in both designs and is taken from r27_per_slice.csv,
                    B and C from r27_revised_per_slice.csv, D is the V2 test set in both designs.
In the revised design the paired comparison of each B/C volume with its original counterpart (same mask, class and
nuisance parameters, different lesion contrast) is also reported as a direct estimate of the low-contrast effect.
"""
import argparse
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

R = "results_revision"
M = ["DSC", "HD", "HD95", "ASD"]
KEY = ["cell", "structure", "appearance", "run", "case_id", "prompt"]
rng = np.random.default_rng(0)
pd.set_option("display.width", 200)


def boot_diff(x, y, paired=False, n=10000):
    if paired:
        d = x - y
        return np.percentile([rng.choice(d, len(d)).mean() for _ in range(n)], [2.5, 97.5])
    return np.percentile([rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(n)], [2.5, 97.5])


def load(design):
    orig = pd.read_csv(f"{R}/r27_per_slice.csv")
    if design == "original":
        cells = orig[KEY + M]
    else:
        rev = pd.read_csv(f"{R}/r27_revised_per_slice.csv")
        cells = pd.concat([orig[orig.cell == "A"][KEY + M], rev[rev.cell.isin(["B", "C"])][KEY + M]], ignore_index=True)
    parts = [cells]
    for prompt, path in (("merged", f"{R}/sam_zeroshot_pad0_per_slice.csv"),
                         ("per_lesion", f"{R}/sam_per_lesion_protocol_per_slice.csv")):
        d = pd.read_csv(path); d["case_id"] = d.file.str.split("_").str[0]
        d = d.assign(cell="D", structure=1, appearance="v2", run="D" + d.run_id.str.replace("run", ""), prompt=prompt)
        parts.append(d[KEY + M])
    return pd.concat(parts, ignore_index=True), orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=["revised", "original"], default="revised")
    a = ap.parse_args()
    df, orig = load(a.design)
    if a.design == "revised":
        names = {"A": "no structure, V1 appearance", "B": "geometry, V1 appearance, std contrast",
                 "C": "no geometry, V2 appearance + low contrast", "D": "geometry, V2 appearance + low contrast (V2)"}
    else:
        names = {"A": "no structure, V1 appearance", "B": "structure, V1 appearance",
                 "C": "no structure, V2 appearance", "D": "structure, V2 appearance (V2)"}
    print(f"design: {a.design}")
    for prompt in ("merged", "per_lesion"):
        sub = df[df.prompt == prompt]; run = sub.groupby(["cell", "run"])[M].mean()
        print(f"\n=== {prompt}: mean ± SD over realizations ===")
        for c in "ABCD":
            r = run.loc[c]
            print(f"{c} {names[c]:44s} n={len(r):2d}  " + "  ".join(f"{k} {r[k].mean():.4f} ± {r[k].std(ddof=1):.4f}" for k in M))
        print("differences of cell means (bootstrap 95% CI over realizations):")
        for a_, b_, lab in (("B", "A", "structure | V1 appearance"), ("D", "C", "structure | V2 appearance"),
                            ("C", "A", "appearance | no structure"), ("D", "B", "appearance | structure")):
            for k in ("DSC", "HD95"):
                x, y = run.loc[a_][k].values, run.loc[b_][k].values
                lo, hi = boot_diff(x, y)
                print(f"  {lab:28s} {k:4s} {x.mean() - y.mean():+8.4f} [{lo:+.4f}, {hi:+.4f}]")
        vol = sub.groupby(["cell", "structure", "appearance", "run", "case_id"]).HD95.mean().reset_index()
        vol["log_hd95"] = np.log1p(vol.HD95); vol["app_v2"] = (vol.appearance == "v2").astype(int)
        fit = smf.mixedlm("log_hd95 ~ structure * app_v2", vol, groups=vol.run).fit(reml=True)
        ci = fit.conf_int(); t = ["structure", "app_v2", "structure:app_v2"]
        out = pd.DataFrame({"effect_x": np.exp(fit.params[t]), "ci_low": np.exp(ci.loc[t, 0]),
                            "ci_high": np.exp(ci.loc[t, 1]), "p": fit.pvalues[t]}).round(4)
        out["p_holm"] = np.minimum(1, np.maximum.accumulate((len(t) - np.arange(len(t))) * np.sort(out.p.values))
                                   )[np.argsort(np.argsort(out.p.values))].round(4)
        print(f"mixed model on log(HD95+1), volumes = {len(vol)} (reference = cell A):"); print(out.to_string())

    if a.design == "revised":
        rev = pd.read_csv(f"{R}/r27_revised_per_slice.csv")
        print("\n=== paired low-contrast effect: same volumes, revised minus original contrast assignment ===")
        print("(B: original had low contrast in ~45% of volumes, revised none; C: the reverse)")
        for cell in ("B", "C"):
            for prompt in ("merged", "per_lesion"):
                o = orig[(orig.cell == cell) & (orig.prompt == prompt)].groupby("run")[M].mean()
                v = rev[(rev.cell == cell) & (rev.prompt == prompt)].groupby("run")[M].mean()
                o, v = o.loc[v.index], v
                for k in ("DSC", "HD95"):
                    d = (v[k] - o[k]).values
                    lo, hi = boot_diff(v[k].values, o[k].values, paired=True)
                    print(f"  cell {cell} {prompt:10s} {k:4s} revised - original {d.mean():+8.4f} [{lo:+.4f}, {hi:+.4f}]"
                          f"  ({(d > 0).sum()}/{len(d)} realizations higher)")


if __name__ == "__main__":
    main()
