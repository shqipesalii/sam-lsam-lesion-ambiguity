import os
import numpy as np, pandas as pd, nibabel as nib
from revision_eval import compute_metrics, PROJECT_DIR

OOD, PRED = f"{PROJECT_DIR}/OOD_datasets", f"{PROJECT_DIR}/OOD_nnunet2d"
rows, fp = [], []
for cell in ("shape", "appearance", "both"):
    for r in range(1, 11):
        d, p = f"{OOD}/{cell}_run{r}", f"{PRED}/{cell}_run{r}"
        if not os.path.isdir(p):
            print("missing", p); continue
        meta = pd.read_csv(f"{d}/metadata.csv")
        for _, m in meta.iterrows():
            gt = nib.load(f"{d}/labelsTs/{m.case_id}.nii.gz").get_fdata() > 0
            pr = nib.load(f"{p}/{m.case_id}.nii.gz").get_fdata() > 0
            if m.cls == "empty":
                fp.append(dict(cell=cell, run_id=f"run{r}", FP=int(pr.any()))); continue
            for z in range(gt.shape[2]):
                if gt[:, :, z].sum() == 0:
                    continue
                dsc, hd, h95, asd, e = compute_metrics(pr[:, :, z], gt[:, :, z])
                rows.append(dict(cell=cell, run_id=f"run{r}", case_id=m.case_id, slice=z, model="nnunet2d",
                                 prompt="none", multi="multiple_lesions" in str(m.variations), variations=m.variations,
                                 DSC=dsc, HD=hd, HD95=h95, ASD=asd, empty_pred=e))
        print(cell, f"run{r} done", flush=True)
df = pd.DataFrame(rows)
df.to_csv(f"{PROJECT_DIR}/lsam_project/results_revision/ood_nnunet2d_per_slice.csv", index=False)
print(df.groupby("cell")[["DSC", "HD95", "ASD", "empty_pred"]].mean().round(4))
f = pd.DataFrame(fp)
print("lesion-free volumes with any foreground:", f.groupby("cell").FP.agg(["sum", "count"]).to_dict("index"))
