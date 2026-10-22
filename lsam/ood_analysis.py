import glob
import numpy as np, pandas as pd
from itertools import combinations
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

R = "results_revision"
META = "/home/shqipe/Desktop/sam_phantom_project/SAM_10_runs/sam_10_runs_all_results.csv"
M = ["DSC", "HD", "HD95", "ASD"]
MODELS = ["sam_zeroshot", "decoder_only", "lsam_full", "nnunet2d"]
CELLS = ["ID", "shape", "appearance", "both"]
rng = np.random.default_rng(0)
pd.set_option("display.width", 220)
meta = pd.read_csv(META, sep=None, engine="python")[["run_id", "file", "variations"]]
meta["multi"] = meta.variations.fillna("").str.contains("multiple_lesions")
fix = lambda s: s.astype(str).str.replace(r"^(\d+)$", r"run\1", regex=True)

def protocol(df, keys):
    df = df.copy(); df["_o"] = (df.prompt == "per_lesion").astype(int)
    return df.sort_values("_o").groupby(keys, as_index=False).last().drop(columns="_o")

idm = []
for m in MODELS[:3]:
    d = pd.read_csv(f"{R}/{m}_pad0_per_slice.csv"); d["model"] = m; idm.append(d)
nn = pd.read_csv(sorted(glob.glob(f"{R}/nnunet2d*per_slice*.csv"))[0]); nn["run_id"] = fix(nn.run_id); nn["model"] = "nnunet2d"
idm.append(nn)
idm = pd.concat(idm, ignore_index=True)[["model", "run_id", "file"] + M].merge(meta, on=["run_id", "file"], how="left")
sam_pl = pd.read_csv(f"{R}/sam_per_lesion_protocol_per_slice.csv").assign(model="sam_zeroshot")
bj = pd.read_csv(f"{R}/box_jitter_per_slice.csv")
bj = protocol(bj[(bj.level == 0) & bj.model.isin(["decoder_only", "lsam_full"])], ["model", "run_id", "file"])
idp = pd.concat([sam_pl[["model", "run_id", "file"] + M], bj[["model", "run_id", "file"] + M],
                 nn[["model", "run_id", "file"] + M]], ignore_index=True).merge(meta, on=["run_id", "file"], how="left")
ID = {"merged": idm.assign(cell="ID"), "per_lesion": idp.assign(cell="ID")}

o = pd.read_csv(f"{R}/ood_per_slice.csv")
onn = pd.read_csv(f"{R}/ood_nnunet2d_per_slice.csv")
keys = ["cell", "model", "run_id", "case_id", "slice"]
OODd = {"merged": pd.concat([o[o.prompt == "merged"], onn], ignore_index=True),
        "per_lesion": pd.concat([protocol(o, keys), onn], ignore_index=True)}

for proto in ("merged", "per_lesion"):
    df = pd.concat([ID[proto], OODd[proto]], ignore_index=True)
    df["multi"] = df.multi.fillna(False).astype(bool)
    run = df.groupby(["model", "cell", "run_id"])[M].mean()
    print(f"\n\n################ protocol: {proto} ################")
    print("\n=== mean ± SD over realizations ===")
    for c in CELLS:
        print(f"-- {c} --")
        for m in MODELS:
            if (m, c) in run.index.droplevel(2):
                r = run.loc[(m, c)]
                print(f"  {m:13s} n={len(r):2d} " + "  ".join(f"{k} {r[k].mean():.4f} ± {r[k].std(ddof=1):.4f}" for k in M))
    print("\n=== change from ID (OOD - ID), bootstrap 95% CI over realizations ===")
    for m in MODELS:
        for c in CELLS[1:]:
            out = []
            for k in ("DSC", "HD95", "ASD"):
                x, y = run.loc[(m, c)][k].values, run.loc[(m, "ID")][k].values
                lo, hi = np.percentile([rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(5000)], [2.5, 97.5])
                out.append(f"{k} {x.mean() - y.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
            print(f"  {m:13s} {c:10s} " + "  ".join(out))
    print("\n=== paired model comparisons within cell (A - B; Wilcoxon exact, Holm over 6 pairs) ===")
    res = []
    for c in CELLS:
        for k in ("DSC", "HD95", "ASD"):
            fam = []
            for a_, b_ in combinations(MODELS, 2):
                x, y = run.loc[(a_, c)][k], run.loc[(b_, c)][k]
                dd = (x - y.reindex(x.index)).dropna().values
                fam.append(dict(cell=c, metric=k, A=a_, B=b_, diff=dd.mean(), A_better_runs=int(((dd > 0) if k == "DSC" else (dd < 0)).sum()),
                                p=wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0))
            ph = multipletests([f["p"] for f in fam], method="holm")[1]
            for f, q in zip(fam, ph):
                f["p_holm"] = q; res.append(f)
    res = pd.DataFrame(res).round(4)
    res.to_csv(f"{R}/ood_comparisons_{proto}.csv", index=False)
    print(res[res.metric == "HD95"].to_string(index=False))
    print("\n=== HD95 by slice subset (mean over realizations) ===")
    sub = df.groupby(["model", "cell", "multi", "run_id"]).HD95.mean().groupby(["model", "cell", "multi"]).mean().unstack("multi").round(4)
    sub.columns = ["single", "multiple"]; print(sub.to_string())
    print("\nempty predictions (missed lesion slices):", df[df.empty_pred == 1].groupby(["model", "cell"]).size().to_dict())
