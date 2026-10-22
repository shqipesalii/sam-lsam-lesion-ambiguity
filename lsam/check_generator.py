import numpy as np, pandas as pd, nibabel as nib
from scipy.ndimage import label
P = "/home/shqipe/Desktop/sam_phantom_project/SAM_10_runs"
rows = []
for r in range(1, 11):
    d = f"{P}/PhantomShapes_V2_run{r}"
    meta = pd.read_csv(f"{d}/metadata.csv")
    for _, m in meta[meta["class"] != "empty"].iterrows():
        lbl = nib.load(f"{d}/labelsTr/{m.case_id}.nii.gz").get_fdata() > 0
        s = lbl[:, :, lbl.shape[2] // 2]; v = str(m.variations)
        rows.append(dict(edge="partial_edge" in v, multi="multiple_lesions" in v, irr="irregular" in v,
                         border=bool(s[0].any() or s[-1].any() or s[:, 0].any() or s[:, -1].any()),
                         ncomp=label(s)[1]))
df = pd.DataFrame(rows); single = df[~df.multi]
print("lesion volumes:", len(df))
print(f"partial-edge touching border: {df[df.edge].border.mean():.1%} (n={df.edge.sum()}); others: {df[~df.edge].border.mean():.1%}")
print(f"single-lesion >1 component: irregular {single[single.irr].ncomp.gt(1).mean():.1%}, regular {single[~single.irr].ncomp.gt(1).mean():.1%}")
print(f"multiple-lesion volumes forming 1 component: {df[df.multi].ncomp.eq(1).mean():.1%}")
