import numpy as np, pandas as pd, nibabel as nib
from scipy.ndimage import binary_dilation, binary_erosion
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

P = "/home/shqipe/Desktop/sam_phantom_project/SAM_10_runs"
R = "results_revision"
rows = []
for r in range(1, 11):
    d = f"{P}/PhantomShapes_V2_run{r}"
    meta = pd.read_csv(f"{d}/metadata.csv")
    for _, m in meta[meta["class"] != "empty"].iterrows():
        img = nib.load(f"{d}/imagesTr/{m.case_id}_0000.nii.gz").get_fdata()
        lbl = nib.load(f"{d}/labelsTr/{m.case_id}.nii.gz").get_fdata() > 0
        vals = []
        for z in range(img.shape[2]):
            les = lbl[:, :, z]
            if les.sum() == 0:
                continue
            core = binary_erosion(les, iterations=1)
            core = core if core.sum() >= 5 else les
            ring = binary_dilation(les, iterations=6) & ~binary_dilation(les, iterations=2) & (img[:, :, z] > -995) & (img[:, :, z] < 995)
            if ring.sum() < 20:
                continue
            x = img[:, :, z]
            mu_l, mu_b, sd_b = x[core].mean(), x[ring].mean(), x[ring].std(ddof=1)
            vals.append((abs(mu_l - mu_b), abs(mu_l - mu_b) / max(sd_b, 1e-6), sd_b))
        if vals:
            v = np.mean(vals, axis=0)
            rows.append(dict(run_id=f"run{r}", case_id=m.case_id, cls=m["class"], variations=m.variations,
                             noise=m.noise_sigma, blur=m.blur_sigma, scale=m.intensity_scale,
                             contrast=v[0], CNR=v[1], bg_sd=v[2]))
    print(f"run{r} done")
cnr = pd.DataFrame(rows)
v = cnr.variations.fillna("")
for f, k in {"irregular": "irregular", "multiple": "multiple_lesions",
             "low_contrast": "low_contrast", "partial_edge": "partial_edge"}.items():
    cnr[f] = v.str.contains(k).astype(int)
cnr.to_csv(f"{R}/cnr_per_volume.csv", index=False)

pd.set_option("display.width", 200)
print(f"\n=== Contrast and CNR by low-contrast flag (lesion volumes = {len(cnr)}) ===")
q = lambda s: f"{s.median():.2f} [{s.quantile(.25):.2f}, {s.quantile(.75):.2f}]  (range {s.min():.2f}-{s.max():.2f})"
for g, sub in cnr.groupby("low_contrast"):
    print(f"low_contrast={g}: n={len(sub)}  contrast {q(sub.contrast)}  CNR {q(sub.CNR)}")
lo, st = cnr[cnr.low_contrast == 1].CNR, cnr[cnr.low_contrast == 0].CNR
print(f"share of low-contrast volumes with CNR above the 25th percentile of standard volumes: {(lo > st.quantile(.25)).mean():.3f}")
print(f"share of standard volumes with CNR below the 75th percentile of low-contrast volumes: {(st < lo.quantile(.75)).mean():.3f}")
print("\nCNR by noise sigma (median):\n", cnr.groupby("noise").CNR.median().round(2).to_string())
print("\nCNR by partial edge (median):", cnr.groupby("partial_edge").CNR.median().round(2).to_dict())

for name, path in (("SAM merged", f"{R}/sam_zeroshot_pad0_per_slice.csv"),
                   ("SAM per-lesion", f"{R}/sam_per_lesion_protocol_per_slice.csv")):
    s = pd.read_csv(path); s["case_id"] = s.file.str.split("_").str[0]
    vol = s.groupby(["run_id", "case_id"]).HD95.mean().reset_index().merge(cnr, on=["run_id", "case_id"])
    vol["log_hd95"] = np.log1p(vol.HD95)
    vol["logCNR_z"] = (np.log(vol.CNR) - np.log(vol.CNR).mean()) / np.log(vol.CNR).std()
    vol["noise_z"] = (vol.noise - vol.noise.mean()) / vol.noise.std()
    form = ("log_hd95 ~ irregular + multiple + partial_edge + logCNR_z + noise_z"
            " + multiple:irregular + multiple:partial_edge + multiple:logCNR_z + blur + scale"
            " + C(cls, Treatment('circle'))")
    fit = smf.mixedlm(form, vol, groups=vol.run_id).fit(reml=True)
    terms = [t for t in fit.params.index if t not in ("Intercept", "Group Var")]
    ci = fit.conf_int()
    out = pd.DataFrame({"effect_x": np.exp(fit.params[terms]), "ci_low": np.exp(ci.loc[terms, 0]),
                        "ci_high": np.exp(ci.loc[terms, 1]), "p_holm": multipletests(fit.pvalues[terms], method="holm")[1]}).round(4)
    print(f"\n=== {name}: factor model with continuous log(CNR) and noise (per 1 SD), volumes = {len(vol)} ===")
    print(out.to_string())
    print("Spearman HD95 vs CNR:", round(vol[["HD95", "CNR"]].corr(method="spearman").iloc[0, 1], 3),
          "| among non-multiple volumes:", round(vol[vol.multiple == 0][["HD95", "CNR"]].corr(method="spearman").iloc[0, 1], 3))
print(f"\nSaved {R}/cnr_per_volume.csv")

cnr["CNR_noise"] = cnr.contrast / cnr.noise
print("\n=== Contrast relative to the imposed noise (contrast / noise sigma) ===")
print(cnr.groupby("noise").CNR_noise.describe(percentiles=[.05, .5]).round(1).to_string())
print("minimum lesion contrast (HU):", round(cnr.contrast.min(), 1), "| 1st percentile:", round(cnr.contrast.quantile(.01), 1))
print("ring SD (HU) median by noise:", cnr.groupby("noise").bg_sd.median().round(1).to_dict())
