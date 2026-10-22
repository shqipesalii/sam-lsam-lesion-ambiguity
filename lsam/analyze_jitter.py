import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

R = "results_revision"
df = pd.read_csv(f"{R}/box_jitter_per_slice.csv")
M = ["DSC", "HD", "HD95", "ASD"]
rng = np.random.default_rng(0)
pd.set_option("display.width", 220)

def per_run(d):
    return d.groupby(["run_id", "file"])[M + ["box_iou"]].mean().groupby("run_id").mean().sort_index()

def sub(model, prompt, level, multi_only=False):
    d = df[(df.model == model) & (df.prompt == prompt) & (df.level == level)]
    return per_run(d[d.multi] if multi_only else d)

def paired(a, b, metric):
    d = (a[metric] - b[metric]).values
    lo, hi = np.percentile([rng.choice(d, len(d)).mean() for _ in range(10000)], [2.5, 97.5])
    favour = (d > 0).sum() if metric == "DSC" else (d < 0).sum()
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return dict(diff=d.mean(), ci_low=lo, ci_high=hi, runs_a_better=int(favour), p=p)

levels = sorted(df.level.unique()); jit = [l for l in levels if l > 0]

try:
    main = pd.read_csv(f"{R}/sam_zeroshot_pad0_per_slice.csv").groupby("run_id")["HD95"].mean().sort_index()
    o = sub("sam_zeroshot", "merged", 0.0)["HD95"]
    print(f"Sanity: oracle SAM HD95 vs main evaluation, max abs diff = {np.abs(o - main.loc[o.index]).max():.4f}")
except FileNotFoundError:
    pass

rows = []
for model in ("sam_zeroshot", "decoder_only", "lsam_full"):
    for prompt, multi, label in (("merged", False, "all slices, merged"), ("merged", True, "multi slices, merged"),
                                 ("per_lesion", True, "multi slices, per-lesion")):
        for l in levels:
            r = sub(model, prompt, l, multi)
            if r.empty:
                continue
            rows.append({"model": model, "setting": label, "level": l, "box_IoU": round(r.box_iou.mean(), 3),
                         **{k: f"{r[k].mean():.4f} ± {r[k].std(ddof=1):.4f}" for k in M}})
perf = pd.DataFrame(rows); perf.to_csv(f"{R}/box_jitter_summary.csv", index=False)
print("\n=== Performance vs jitter level (mean ± SD over 10 realizations) ===")
for s in perf.setting.unique():
    print(f"\n-- {s} --"); print(perf[perf.setting == s].drop(columns="setting").to_string(index=False))

def family(name, pairs, metrics=("DSC", "HD95")):
    out = []
    for metric in metrics:
        fam = [dict(comparison=name, metric=metric, level=l, **paired(a, b, metric)) for l, a, b in pairs]
        for f, p in zip(fam, multipletests([f["p"] for f in fam], method="holm")[1]):
            f["p_holm"] = p
        out += fam
    return out

res = []
for model in ("sam_zeroshot", "decoder_only", "lsam_full"):
    o = sub(model, "merged", 0.0)
    res += family(f"{model}: jitter - oracle", [(l, sub(model, "merged", l), o) for l in jit])
res += family("SAM multi: per_lesion - merged",
              [(l, sub("sam_zeroshot", "per_lesion", l, True), sub("sam_zeroshot", "merged", l, True)) for l in levels])
for a_, b_ in (("decoder_only", "sam_zeroshot"), ("lsam_full", "sam_zeroshot"), ("lsam_full", "decoder_only")):
    res += family(f"{a_} - {b_}", [(l, sub(a_, "merged", l), sub(b_, "merged", l)) for l in levels])
    res += family(f"{a_} - {b_} (multi slices)", [(l, sub(a_, "merged", l, True), sub(b_, "merged", l, True)) for l in levels])
res = pd.DataFrame(res)
res[["diff", "ci_low", "ci_high", "p", "p_holm"]] = res[["diff", "ci_low", "ci_high", "p", "p_holm"]].round(4)
res.to_csv(f"{R}/box_jitter_paired.csv", index=False)
print("\n=== Paired differences (A - B; runs_a_better of 10; Holm over levels) ===")
for c in res.comparison.unique():
    print(f"\n-- {c} --"); print(res[res.comparison == c].drop(columns="comparison").to_string(index=False))

print("\n=== Relative change vs oracle box (merged prompt, all slices) ===")
for model in ("sam_zeroshot", "decoder_only", "lsam_full"):
    o = sub(model, "merged", 0.0)
    for l in jit:
        r = sub(model, "merged", l)
        print(f"{model:13s} level {l:.2f}: DSC {100*(r.DSC.mean()/o.DSC.mean()-1):+6.1f}%   "
              f"HD95 {100*(r.HD95.mean()/o.HD95.mean()-1):+7.1f}%   ASD {100*(r.ASD.mean()/o.ASD.mean()-1):+7.1f}%")
