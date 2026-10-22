import glob, numpy as np, nibabel as nib
from medpy.metric.binary import dc, hd, hd95, assd
N = "/home/shqipe/Desktop/nnunet_project"
PEN = 362.04
rows = []
for r in range(1, 11):
    ds = f"Dataset5{r:02d}_PhantomShapesV2Run{r}"
    vd = glob.glob(f"{N}/nnUNet_results/{ds}/*3d_fullres*/fold_0/validation")
    if not vd: print(f"run{r}: validation folder not found"); continue
    for pf in sorted(glob.glob(vd[0] + "/*.nii.gz")):
        case = pf.split("/")[-1]
        gf = f"{N}/nnUNet_preprocessed/{ds}/gt_segmentations/{case}"
        g = nib.load(gf).get_fdata() > 0; p = nib.load(pf).get_fdata() > 0
        if not g.any(): continue                      # lesion-free volume
        if not p.any(): rows.append((r, 0.0, PEN, PEN, PEN, 1)); continue
        rows.append((r, dc(p, g), hd(p, g), hd95(p, g), assd(p, g), 0))
a = np.array(rows)
print(f"lesion volumes: {len(a)}, empty predictions: {int(a[:,5].sum())}")
ne = a[a[:,5] == 0]
for name, d in (("WITH penalty (paper's rule)", a), ("EXCLUDING empty predictions", ne)):
    print(f"{name:28s} DSC {d[:,1].mean():.4f} ± {d[:,1].std(ddof=1):.4f}   HD {d[:,2].mean():.2f} ± {d[:,2].std(ddof=1):.2f}   "
          f"HD95 {d[:,3].mean():.2f} ± {d[:,3].std(ddof=1):.2f}   ASD {d[:,4].mean():.3f} ± {d[:,4].std(ddof=1):.3f}")
