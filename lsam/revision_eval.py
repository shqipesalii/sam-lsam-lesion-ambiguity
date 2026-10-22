import os
import argparse
import random
from glob import glob

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from medpy.metric.binary import dc, hd as medpy_hd, hd95 as medpy_hd95, assd as medpy_assd

PROJECT_DIR = "/home/shqipe/Desktop/sam_phantom_project"
SAM_CHECKPOINT = f"{PROJECT_DIR}/checkpoints/sam_vit_b_01ec64.pth"
CKPT_DIR = f"{PROJECT_DIR}/lsam_project/checkpoints"
OUT_DIR = f"{PROJECT_DIR}/lsam_project/results_revision"
IMG_SIZE = 1024
EMPTY_DIST = float(np.hypot(256, 256))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0); random.seed(0)


class IdentityBlock(nn.Module):
    def forward(self, x):
        return x


class LowContrastEnhancementBlock(nn.Module):
    def __init__(self, c=256):
        super().__init__()
        self.attention = nn.Sequential(nn.Conv2d(c, c, 1), nn.Sigmoid())

    def forward(self, x):
        return x + x * self.attention(x)


class IrregularBoundaryBlock(nn.Module):
    def __init__(self, c=256):
        super().__init__()
        self.edge_refine = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
                                         nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.edge_refine(x)


class MultiLesionContextBlock(nn.Module):
    def __init__(self, c=256):
        super().__init__()
        self.context = nn.Sequential(nn.Conv2d(c, c, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
                                     nn.Conv2d(c, c, 1))

    def forward(self, x):
        return x + self.context(x)


class SimpleAdapter(nn.Module):
    def __init__(self, c=256):
        super().__init__()
        self.adapter = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(c, c, 1))

    def forward(self, x):
        return x + self.adapter(x)


class AblationAdapter(nn.Module):
    def __init__(self, variant, c=256):
        super().__init__()
        if variant == "simple_adapter":
            self.adapter = SimpleAdapter(c)
        else:
            self.low_contrast = IdentityBlock() if variant == "no_low_contrast" else LowContrastEnhancementBlock(c)
            self.boundary_modeling = IdentityBlock() if variant == "no_boundary_block" else IrregularBoundaryBlock(c)
            self.multi_lesion_context = IdentityBlock() if variant == "no_multi_context" else MultiLesionContextBlock(c)
            self.adapter = None

    def forward(self, x):
        if self.adapter is not None:
            return self.adapter(x)
        return self.multi_lesion_context(self.boundary_modeling(self.low_contrast(x)))


class FullAdapter(nn.Module):
    def __init__(self, c=256):
        super().__init__()
        self.low_contrast = LowContrastEnhancementBlock(c)
        self.boundary_modeling = IrregularBoundaryBlock(c)
        self.multi_lesion_context = MultiLesionContextBlock(c)

    def forward(self, x):
        return self.multi_lesion_context(self.boundary_modeling(self.low_contrast(x)))


def get_box(mask, pad):
    ys, xs = np.where(mask > 0)
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(mask.shape[1] - 1, x2 + pad), min(mask.shape[0] - 1, y2 + pad)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def compute_metrics(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if pred.sum() == 0:
        return 0.0, EMPTY_DIST, EMPTY_DIST, EMPTY_DIST, 1
    return dc(pred, gt), medpy_hd(pred, gt), medpy_hd95(pred, gt), medpy_assd(pred, gt), 0


def load_model(name, ckpt_path):
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT).to(DEVICE).eval()
    ck = torch.load(ckpt_path, map_location=DEVICE)
    sam.mask_decoder.load_state_dict(ck["sam_mask_decoder"])
    if name == "decoder_only":
        adapter = IdentityBlock()
    elif name == "lsam_full":
        adapter = FullAdapter(); adapter.load_state_dict(ck["adapter"])
    else:
        adapter = AblationAdapter(name); adapter.load_state_dict(ck["adapter"])
    return sam, adapter.to(DEVICE).eval()


@torch.no_grad()
def evaluate_sam_zeroshot(runs, pad):
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT).to(DEVICE).eval()
    predictor = SamPredictor(sam)
    rows = []
    for run_id in runs:
        img_dir = f"{PROJECT_DIR}/SAM_10_runs/SAM_dataset_run{run_id}/images"
        msk_dir = f"{PROJECT_DIR}/SAM_10_runs/SAM_dataset_run{run_id}/masks"
        for img_path in tqdm(sorted(glob(os.path.join(img_dir, "*.png"))), desc=f"sam_zeroshot run{run_id}"):
            fname = os.path.basename(img_path)
            gt = (cv2.imread(os.path.join(msk_dir, fname), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            if gt.sum() == 0:
                continue
            predictor.set_image(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
            masks, scores, _ = predictor.predict(box=get_box(gt, pad), multimask_output=True)
            pred = masks[int(np.argmax(scores))].astype(np.uint8)
            d, h, h95, s, e = compute_metrics(pred, gt)
            rows.append(dict(model="sam_zeroshot", pad=pad, run_id=f"run{run_id}", file=fname,
                             DSC=d, HD=h, HD95=h95, ASD=s, empty_pred=e))
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate(name, sam, adapter, runs, pad):
    rows = []
    for run_id in runs:
        img_dir = f"{PROJECT_DIR}/SAM_10_runs/SAM_dataset_run{run_id}/images"
        msk_dir = f"{PROJECT_DIR}/SAM_10_runs/SAM_dataset_run{run_id}/masks"
        for img_path in tqdm(sorted(glob(os.path.join(img_dir, "*.png"))), desc=f"{name} run{run_id}"):
            fname = os.path.basename(img_path)
            mpath = os.path.join(msk_dir, fname)
            if not os.path.exists(mpath):
                continue
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            gt = (cv2.imread(mpath, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            if gt.sum() == 0:
                continue
            h, w = image.shape[:2]
            box = get_box(gt, pad) * np.array([IMG_SIZE / w, IMG_SIZE / h, IMG_SIZE / w, IMG_SIZE / h], dtype=np.float32)
            x = torch.tensor(cv2.resize(image, (IMG_SIZE, IMG_SIZE))).permute(2, 0, 1)[None].float().to(DEVICE)
            emb = adapter(sam.image_encoder(sam.preprocess(x)))
            sp, de = sam.prompt_encoder(points=None, boxes=torch.tensor(box)[None].to(DEVICE), masks=None)
            low, _ = sam.mask_decoder(image_embeddings=emb, image_pe=sam.prompt_encoder.get_dense_pe(),
                                      sparse_prompt_embeddings=sp, dense_prompt_embeddings=de,
                                      multimask_output=False)
            pred = (torch.sigmoid(low).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
            d, hh, h95, s, e = compute_metrics(pred, gt)
            rows.append(dict(model=name, pad=pad, run_id=f"run{run_id}", file=fname,
                             DSC=d, HD=hh, HD95=h95, ASD=s, empty_pred=e))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder_only_ckpt", required=True)
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--runs", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--models", nargs="+", default=None)
    a = ap.parse_args()
    models = {
        "sam_zeroshot": SAM_CHECKPOINT,
        "lsam_full": f"{CKPT_DIR}/lsam_full_best.pth",
        "decoder_only": a.decoder_only_ckpt,
        "ablation_full": f"{CKPT_DIR}/lsam_ablation_full.pth",
        "no_boundary_loss": f"{CKPT_DIR}/lsam_ablation_no_boundary_loss.pth",
        "no_low_contrast": f"{CKPT_DIR}/lsam_ablation_no_low_contrast.pth",
        "no_boundary_block": f"{CKPT_DIR}/lsam_ablation_no_boundary_block.pth",
        "no_multi_context": f"{CKPT_DIR}/lsam_ablation_no_multi_context.pth",
        "simple_adapter": f"{CKPT_DIR}/lsam_ablation_simple_adapter.pth",
    }
    if a.models:
        models = {k: v for k, v in models.items() if k in a.models}
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Device:", DEVICE, "| pad =", a.pad)
    summaries = []
    for name, path in models.items():
        if not os.path.exists(path):
            print(f"SKIP {name}: checkpoint not found at {path}"); continue
        if name == "sam_zeroshot":
            df = evaluate_sam_zeroshot(a.runs, a.pad); sam = adapter = None
        else:
            sam, adapter = load_model("full" if name == "ablation_full" else name, path)
            df = evaluate(name, sam, adapter, a.runs, a.pad)
        df.to_csv(f"{OUT_DIR}/{name}_pad{a.pad}_per_slice.csv", index=False)
        s = df.groupby("run_id").agg(DSC=("DSC", "mean"), HD=("HD", "mean"), HD95=("HD95", "mean"),
                                     ASD=("ASD", "mean"), empty_pred=("empty_pred", "sum")).reset_index()
        s.insert(0, "model", name); summaries.append(s)
        r = s[["DSC", "HD", "HD95", "ASD"]].agg(["mean", "std"])
        print(f"\n{name} (pad={a.pad}), slices={len(df)}, empty={int(df.empty_pred.sum())}")
        for m in ["DSC", "HD", "HD95", "ASD"]:
            print(f"  {m}: {r.loc['mean', m]:.4f} +/- {r.loc['std', m]:.4f}")
        del sam, adapter; torch.cuda.empty_cache()
    if summaries:
        out = f"{OUT_DIR}/all_models_pad{a.pad}_per_run.csv"
        pd.concat(summaries).to_csv(out, index=False); print("\nSaved:", out)


if __name__ == "__main__":
    main()
