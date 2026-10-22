import os, glob
import numpy as np, pandas as pd, nibabel as nib
from medpy.metric.binary import dc, hd, hd95, assd

P = "/home/shqipe/Desktop/sam_phantom_project"
EMPTY = float(np.hypot(256, 256))
rows = []
for r in range(1, 11):
    pred_dir = f"{P}/nnunet_2d_project/predictions/nnunet2d_run{r}"
    gt_dir = f"{P}/SAM_10_runs/PhantomShapes_V2_run{r}/labelsTr"
    for gf in sorted(glob.glob(f"{gt_dir}/Case*.nii.gz")):
        case = os.path.basename(gf).replace(".nii.gz", "")
        pf = f"{pred_dir}/{case}.nii.gz"
        if not os.path.exists(pf):
            continue
        g = nib.load(gf).get_fdata() > 0
        p = nib.load(pf).get_fdata() > 0
        for z in range(g.shape[-1]):
            gs, ps = g[..., z].astype(np.uint8), p[..., z].astype(np.uint8)
            if gs.sum() == 0:
                continue                      # lesion-containing slices only, as for SAM/LSAM
            if ps.sum() == 0:
                m = (0.0, EMPTY, EMPTY, EMPTY, 1)
            else:
                m = (dc(ps, gs), hd(ps, gs), hd95(ps, gs), assd(ps, gs), 0)
            rows.append(dict(model="nnunet2d", run_id=f"run{r}", file=f"{case}_slice{z:03d}.png",
                             DSC=m[0], HD=m[1], HD95=m[2], ASD=m[3], empty_pred=m[4]))
df = pd.DataFrame(rows)
os.makedirs("results_revision", exist_ok=True)
df.to_csv("results_revision/nnunet2d_medpy_per_slice.csv", index=False)
s = df.groupby("run_id").agg(DSC=("DSC", "mean"), HD=("HD", "mean"), HD95=("HD95", "mean"),
                             ASD=("ASD", "mean"), empty_pred=("empty_pred", "sum")).reset_index()
s.insert(0, "model", "nnunet2d")
s.to_csv("results_revision/nnunet2d_medpy_per_run.csv", index=False)
r = s[["DSC", "HD", "HD95", "ASD"]].agg(["mean", "std"])
print(f"nnU-Net 2D (MedPy): slices={len(df)}, empty={int(df.empty_pred.sum())}")
for m in ["DSC", "HD", "HD95", "ASD"]:
    print(f"  {m}: {r.loc['mean', m]:.4f} +/- {r.loc['std', m]:.4f}")
