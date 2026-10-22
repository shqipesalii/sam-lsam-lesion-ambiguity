import numpy as np, pandas as pd
from itertools import combinations
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

R = "results_revision"
M = ["DSC", "HD", "HD95", "ASD"]
MODELS = ["sam_zeroshot", "decoder_only", "lsam_full"]
rng = np.random.default_rng(0)
pd.set_option("display.width", 220)
df = pd.read_csv(f"{R}/busi_per_image.csv")
import os
if os.path.exists(f"{R}/busi_medsam_per_image.csv"):
    df = pd.concat([df, pd.read_csv(f"{R}/busi_medsam_per_image.csv")], ignore_index=True)
    MODELS = MODELS + ["medsam"]
if os.path.exists(f"{R}/busi_sam2_per_image.csv"):
    df = pd.concat([df, pd.read_csv(f"{R}/busi_sam2_per_image.csv")], ignore_index=True)
    MODELS = MODELS + ["sam2"]
img = df.groupby(["file", "cls", "dup", "n_lesions", "model", "prompt", "level"])[M + ["empty_pred"]].mean().reset_index()
proto = {"merged": img[img.prompt == "merged"],
         "per_lesion": img.assign(_o=(img.prompt == "per_lesion").astype(int)).sort_values("_o")
                           .groupby(["file", "model", "level"], as_index=False).last()}
print(f"lesion images: {img.file.nunique()} | near-duplicates flagged: {img.drop_duplicates('file').dup.sum()} | "
      f"multi-lesion images: {(img.drop_duplicates('file').n_lesions > 1).sum()} | classes:",
      img.drop_duplicates("file").cls.value_counts().to_dict())
for subset, keep in (("all lesion images", lambda d: d), ("near-duplicates removed", lambda d: d[d.dup == 0])):
    print(f"\n\n################ {subset} ################")
    for pname, d0 in proto.items():
        d0 = keep(d0)
        print(f"\n=== protocol {pname}: mean ± SD and median [IQR] over images ===")
        for lv in sorted(d0.level.unique()):
            for m in MODELS:
                s = d0[(d0.level == lv) & (d0.model == m)]
                print(f"  box ±{lv:.0%} {m:13s} n={len(s)} " + "  ".join(
                    f"{k} {s[k].mean():.3f} ± {s[k].std():.3f} (med {s[k].median():.3f} [{s[k].quantile(.25):.3f}, {s[k].quantile(.75):.3f}])"
                    for k in ("DSC", "HD95", "ASD")) + f"  empty {int(s.empty_pred.gt(0).sum())}")
        print(f"--- by class (exact box, {pname}) ---")
        print(d0[d0.level == 0].groupby(["cls", "model"])[["DSC", "HD95", "ASD"]].mean().round(3).to_string())
        print(f"--- paired comparisons (A - B), Holm over 3 pairs ---")
        out = []
        for lv in sorted(d0.level.unique()):
            for k in ("DSC", "HD95", "ASD"):
                piv = d0[d0.level == lv].pivot_table(index="file", columns="model", values=k)
                fam = []
                for a_, b_ in combinations(MODELS, 2):
                    dd = (piv[a_] - piv[b_]).dropna().values
                    boot = [rng.choice(dd, len(dd)).mean() for _ in range(5000)]
                    fam.append(dict(level=lv, metric=k, A=a_, B=b_, diff=dd.mean(),
                                    ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5),
                                    A_better_share=((dd > 0) if k == "DSC" else (dd < 0)).mean(),
                                    p=wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0))
                for f, q in zip(fam, multipletests([f["p"] for f in fam], method="holm")[1]):
                    f["p_holm"] = q; out.append(f)
        out = pd.DataFrame(out).round(4)
        out.to_csv(f"{R}/busi_comparisons_{pname}_{'all' if subset.startswith('all') else 'dedup'}.csv", index=False)
        print(out[out.metric.isin(["DSC", "HD95"])].to_string(index=False))
