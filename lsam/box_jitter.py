"""
R2.3: box-jitter robustness. Each box coordinate shifted independently by U(-d,d)*side,
d in {0, 0.05, 0.10, 0.20}, 5 repetitions per slice, identical boxes for all models.
Models: zero-shot SAM, decoder-only, LSAM. Prompts: merged (all slices) + per-lesion (multi slices).
"""
import os, zlib, argparse
from glob import glob
import cv2, numpy as np, pandas as pd, torch
from scipy import ndimage
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
import medpy.metric.binary as mb
from revision_eval import load_model, PROJECT_DIR, SAM_CHECKPOINT, IMG_SIZE, DEVICE, EMPTY_DIST

OUT_DIR = f"{PROJECT_DIR}/lsam_project/results_revision"
META_CSV = f"{PROJECT_DIR}/SAM_10_runs/sam_10_runs_all_results.csv"
_surf = getattr(mb, "__surface_distances", None)
_checked = {"n": 0}

def metrics(pred, gt):
    pred = (pred > 0).astype(np.uint8); gt = (gt > 0).astype(np.uint8)
    if pred.sum() == 0:
        return 0.0, EMPTY_DIST, EMPTY_DIST, EMPTY_DIST, 1
    d = mb.dc(pred, gt)
    if _surf is None:
        return d, mb.hd(pred, gt), mb.hd95(pred, gt), mb.assd(pred, gt), 0
    a, b = _surf(pred, gt, None, 1), _surf(gt, pred, None, 1)
    h, h95, s = max(a.max(), b.max()), np.percentile(np.hstack((a, b)), 95), np.hstack((a, b)).mean()
    if _checked["n"] < 50:
        ref = (mb.hd(pred, gt), mb.hd95(pred, gt), mb.assd(pred, gt))
        assert np.allclose((h, h95, s), ref), f"fast metrics differ from MedPy: {(h, h95, s)} vs {ref}"
        _checked["n"] += 1
    return d, h, h95, s, 0

def tight_box(mask):
    ys, xs = np.where(mask > 0)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

def jitter(box, level, seed, size=256):
    if level == 0:
        return box.copy()
    rng = np.random.default_rng(seed)
    w, h = box[2] - box[0] + 1, box[3] - box[1] + 1
    b = box + rng.uniform(-level, level, 4) * np.array([w, h, w, h], dtype=np.float32)
    b = np.clip(b, 0, size - 1)
    if b[2] < b[0] + 1: b[0], b[2] = sorted((b[0], b[2])); b[2] = min(size - 1, b[0] + 1)
    if b[3] < b[1] + 1: b[1], b[3] = sorted((b[1], b[3])); b[3] = min(size - 1, b[1] + 1)
    return b.astype(np.float32)

def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def seed_of(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())

@torch.no_grad()
def finetuned_mask(sam, adapter, emb_raw, box, w, h):
    emb = adapter(emb_raw)
    b = box * np.array([IMG_SIZE / w, IMG_SIZE / h, IMG_SIZE / w, IMG_SIZE / h], dtype=np.float32)
    sp, de = sam.prompt_encoder(points=None, boxes=torch.tensor(b)[None].to(DEVICE), masks=None)
    low, _ = sam.mask_decoder(image_embeddings=emb, image_pe=sam.prompt_encoder.get_dense_pe(),
                              sparse_prompt_embeddings=sp, dense_prompt_embeddings=de, multimask_output=False)
    pred = (torch.sigmoid(low).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder_only_ckpt", required=True)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--runs", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    meta = pd.read_csv(META_CSV, sep=None, engine="python")
    mm = meta[meta.variations.fillna("").str.contains("multiple_lesions")]
    multi_files = set(zip(mm.run_id, mm.file))

    sam0 = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT).to(DEVICE).eval()
    predictor = SamPredictor(sam0)
    ft = {"decoder_only": load_model("decoder_only", a.decoder_only_ckpt),
          "lsam_full": load_model("lsam_full", f"{PROJECT_DIR}/lsam_project/checkpoints/lsam_full_best.pth")}
    enc = ft["lsam_full"][0]

    rows = []
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = f"{OUT_DIR}/box_jitter_per_slice.csv"
    for r in a.runs:
        img_dir = f"{PROJECT_DIR}/SAM_10_runs/SAM_dataset_run{r}/images"
        msk_dir = f"{PROJECT_DIR}/SAM_10_runs/SAM_dataset_run{r}/masks"
        files = sorted(glob(os.path.join(img_dir, "*.png")))
        if a.limit:
            files = files[:a.limit]
        for p in tqdm(files, desc=f"run{r}"):
            f = os.path.basename(p)
            mp = os.path.join(msk_dir, f)
            if not os.path.exists(mp):
                continue
            gt = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            if gt.sum() == 0:
                continue
            image = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            is_multi = (f"run{r}", f) in multi_files
            labels, n_comp = ndimage.label(gt)
            with torch.no_grad():
                predictor.set_image(image)
                x = torch.tensor(cv2.resize(image, (IMG_SIZE, IMG_SIZE))).permute(2, 0, 1)[None].float().to(DEVICE)
                emb_raw = enc.image_encoder(enc.preprocess(x))

            def predict(model, box):
                if model == "sam_zeroshot":
                    m, s, _ = predictor.predict(box=box, multimask_output=True)
                    return m[int(np.argmax(s))].astype(np.uint8)
                sam, ad = ft[model]
                return finetuned_mask(sam, ad, emb_raw, box, w, h)

            prompts = [("merged", [tight_box(gt)])]
            if is_multi:
                prompts.append(("per_lesion", [tight_box(labels == k) for k in range(1, n_comp + 1)]))
            for level in a.levels:
                for rep in ([0] if level == 0 else range(a.reps)):
                    for pname, oracle in prompts:
                        boxes = [jitter(b, level, seed_of(r, f, level, rep, pname, i), w) for i, b in enumerate(oracle)]
                        iou = float(np.mean([box_iou(b, o) for b, o in zip(boxes, oracle)]))
                        for model in ("sam_zeroshot", "decoder_only", "lsam_full"):
                            pred = np.zeros_like(gt)
                            for b in boxes:
                                pred |= predict(model, b)
                            d, hd, h95, s, e = metrics(pred, gt)
                            rows.append(dict(model=model, prompt=pname, level=level, rep=rep, run_id=f"run{r}",
                                             file=f, multi=is_multi, n_components=n_comp, box_iou=iou,
                                             DSC=d, HD=hd, HD95=h95, ASD=s, empty_pred=e))
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv, "| rows:", len(df), "| empty predictions:", int(df.empty_pred.sum()))
    s = (df[df.prompt == "merged"].groupby(["model", "level", "run_id"])[["DSC", "HD95", "ASD", "box_iou"]].mean()
         .groupby(["model", "level"]).mean().round(4))
    print("\nMerged-box prompt, mean over realizations:\n", s.to_string())

if __name__ == "__main__":
    main()
