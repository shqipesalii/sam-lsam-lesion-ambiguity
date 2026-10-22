import os
from glob import glob
import cv2, numpy as np, pandas as pd, torch
from tqdm import tqdm
from revision_eval import load_model, PROJECT_DIR, DEVICE, IMG_SIZE
from box_jitter import metrics, tight_box, jitter, seed_of, finetuned_mask

CK = f"{PROJECT_DIR}/lsam_project/checkpoints"
MODELS = {"ablation_full": ("full", f"{CK}/lsam_ablation_full.pth"),
          "no_boundary_loss": ("no_boundary_loss", f"{CK}/lsam_ablation_no_boundary_loss.pth"),
          "no_low_contrast": ("no_low_contrast", f"{CK}/lsam_ablation_no_low_contrast.pth"),
          "no_boundary_block": ("no_boundary_block", f"{CK}/lsam_ablation_no_boundary_block.pth"),
          "no_multi_context": ("no_multi_context", f"{CK}/lsam_ablation_no_multi_context.pth"),
          "simple_adapter": ("simple_adapter", f"{CK}/lsam_ablation_simple_adapter.pth")}
LEVELS, REPS = [0.0, 0.05, 0.10, 0.20], 5
OUT = f"{PROJECT_DIR}/lsam_project/results_revision/box_jitter_ablation_per_slice.csv"

nets = {k: load_model(arch, p) for k, (arch, p) in MODELS.items()}
enc = next(iter(nets.values()))[0]
rows = []
for r in range(1, 11):
    img_dir = f"{PROJECT_DIR}/SAM_10_runs/SAM_dataset_run{r}/images"
    msk_dir = f"{PROJECT_DIR}/SAM_10_runs/SAM_dataset_run{r}/masks"
    for p in tqdm(sorted(glob(os.path.join(img_dir, "*.png"))), desc=f"run{r}"):
        f = os.path.basename(p); mp = os.path.join(msk_dir, f)
        if not os.path.exists(mp):
            continue
        gt = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        if gt.sum() == 0:
            continue
        image = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB); h, w = image.shape[:2]
        with torch.no_grad():
            x = torch.tensor(cv2.resize(image, (IMG_SIZE, IMG_SIZE))).permute(2, 0, 1)[None].float().to(DEVICE)
            emb_raw = enc.image_encoder(enc.preprocess(x))
        oracle = tight_box(gt)
        for level in LEVELS:
            for rep in ([0] if level == 0 else range(REPS)):
                box = jitter(oracle, level, seed_of(r, f, level, rep, "merged", 0), w)
                for name, (sam, ad) in nets.items():
                    d, hd, h95, s, e = metrics(finetuned_mask(sam, ad, emb_raw, box, w, h), gt)
                    rows.append(dict(model=name, level=level, rep=rep, run_id=f"run{r}", file=f,
                                     DSC=d, HD=hd, HD95=h95, ASD=s, empty_pred=e))
    pd.DataFrame(rows).to_csv(OUT, index=False)
print("Saved:", OUT, "| rows:", len(rows))
