import os
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
from scipy import ndimage
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from medpy.metric.binary import dc, hd as medpy_hd, hd95 as medpy_hd95, assd as medpy_assd

PROJECT_DIR = "/home/shqipe/Desktop/sam_phantom_project"
SAM_CHECKPOINT = f"{PROJECT_DIR}/checkpoints/sam_vit_b_01ec64.pth"
META_CSV = f"{PROJECT_DIR}/SAM_10_runs/sam_10_runs_all_results.csv"
OUT_DIR = f"{PROJECT_DIR}/lsam_project/results_revision"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMPTY_DIST = float(np.hypot(256, 256))

def metrics(pred, gt):
    pred = (pred > 0).astype(np.uint8); gt = (gt > 0).astype(np.uint8)
    if pred.sum() == 0:
        return 0.0, EMPTY_DIST, EMPTY_DIST, EMPTY_DIST
    return dc(pred, gt), medpy_hd(pred, gt), medpy_hd95(pred, gt), medpy_assd(pred, gt)


def tight_box(mask):
    ys, xs = np.where(mask > 0)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def best_mask(predictor, box):
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    return masks[int(np.argmax(scores))].astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    meta = pd.read_csv(META_CSV, sep=None, engine="python")
    multi = meta[meta["variations"].fillna("").str.contains("multiple_lesions")]
    print("Multiple-lesion slices in metadata:", len(multi))

    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT).to(DEVICE).eval()
    predictor = SamPredictor(sam)
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    for r in a.runs:
        sub = multi[multi["run_id"] == f"run{r}"]
        if a.limit:
            sub = sub.head(a.limit)
        img_dir = f"{PROJECT_DIR}/SAM_10_runs/SAM_dataset_run{r}/images"
        msk_dir = f"{PROJECT_DIR}/SAM_10_runs/SAM_dataset_run{r}/masks"
        for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"run{r}"):
            f = row["file"]
            image = cv2.cvtColor(cv2.imread(os.path.join(img_dir, f)), cv2.COLOR_BGR2RGB)
            gt = (cv2.imread(os.path.join(msk_dir, f), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            labels, n_comp = ndimage.label(gt)
            with torch.no_grad():
                predictor.set_image(image)
                merged = best_mask(predictor, tight_box(gt))
                per = np.zeros_like(gt)
                for k in range(1, n_comp + 1):
                    per |= best_mask(predictor, tight_box(labels == k))
            for setting, pred in (("merged", merged), ("per_lesion", per)):
                d, h, h95, s = metrics(pred, gt)
                rows.append(dict(run_id=f"run{r}", file=f, case_id=row["case_id"],
                                 variations=row["variations"], n_components=n_comp,
                                 setting=setting, DSC=d, HD=h, HD95=h95, ASD=s,
                                 orig_DSC=row["DSC"], orig_HD95=row["HD95"]))

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/per_lesion_vs_merged_per_slice.csv", index=False)
    run = df.groupby(["run_id", "setting"])[["DSC", "HD", "HD95", "ASD"]].mean().unstack("setting")
    run.to_csv(f"{OUT_DIR}/per_lesion_vs_merged_per_run.csv")

    print("\nConnected components per multiple-lesion slice:",
          df[df.setting == "merged"].n_components.value_counts().sort_index().to_dict())
    chk = df[df.setting == "merged"]
    print("Merged-box re-run vs original HD95 (should be ~0): mean abs diff =",
          round((chk.HD95 - chk.orig_HD95).abs().mean(), 4))
    for m in ["DSC", "HD", "HD95", "ASD"]:
        mg, pl = run[(m, "merged")], run[(m, "per_lesion")]
        print(f"{m:5s} merged {mg.mean():8.4f} +/- {mg.std():.4f}   per-lesion {pl.mean():8.4f} +/- {pl.std():.4f}")
    print("\nSaved:", f"{OUT_DIR}/per_lesion_vs_merged_per_slice.csv")


if __name__ == "__main__":
    main()
