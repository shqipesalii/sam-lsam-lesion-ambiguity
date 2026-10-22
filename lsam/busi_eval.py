import os, re, glob, zlib, argparse
from collections import defaultdict
import numpy as np, pandas as pd, cv2, torch
from medpy.metric.binary import dc, hd, hd95, assd
from segment_anything import sam_model_registry, SamPredictor
from revision_eval import load_model, PROJECT_DIR, SAM_CHECKPOINT, IMG_SIZE, DEVICE
from box_jitter import finetuned_mask, tight_box

OUT = f"{PROJECT_DIR}/lsam_project/results_revision/busi_per_image.csv"
CKPT = f"{PROJECT_DIR}/lsam_project/checkpoints"

def find_pairs(root):
    pairs = defaultdict(list)
    for m in glob.glob(os.path.join(root, "**", "*.png"), recursive=True):
        b = os.path.basename(m)
        if "_mask" in b:
            img = os.path.join(os.path.dirname(m), re.sub(r"_mask(_\d+)?\.png$", ".png", b))
            for cand in (img, img.replace(os.sep + "masks" + os.sep, os.sep + "images" + os.sep)):
                if os.path.exists(cand):
                    pairs[cand].append(m); break
        elif os.sep + "masks" + os.sep in m:
            cand = m.replace(os.sep + "masks" + os.sep, os.sep + "images" + os.sep)
            if os.path.exists(cand):
                pairs[cand].append(m)
    return pairs

def cls_of(path):
    p = os.path.basename(path).lower()
    return "malignant" if "malignant" in p else "benign" if "benign" in p else "normal" if "normal" in p else "unknown"

def flag_duplicates(paths, thr=0.97):
    v = np.stack([cv2.resize(cv2.imread(p, 0), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
                  for p in paths])
    v = (v - v.mean(1, keepdims=True)) / (v.std(1, keepdims=True) + 1e-6)
    c = v @ v.T / v.shape[1]
    dup = np.zeros(len(paths), int)
    for i in range(len(paths)):
        if dup[i]:
            continue
        j = np.where(c[i, i + 1:] > thr)[0] + i + 1
        dup[j] = 1
    return dup

def metrics(p, g):
    p, g = (p > 0).astype(np.uint8), (g > 0).astype(np.uint8)
    if p.sum() == 0:
        e = float(np.hypot(*g.shape)); return 0.0, e, e, e, 1
    return dc(p, g), hd(p, g), hd95(p, g), assd(p, g), 0

def jitter(box, level, seed, w, h):
    if level == 0:
        return box.copy()
    rng = np.random.default_rng(seed)
    bw, bh = box[2] - box[0] + 1, box[3] - box[1] + 1
    b = box + rng.uniform(-level, level, 4) * np.array([bw, bh, bw, bh], np.float32)
    b = np.clip(b, 0, [w - 1, h - 1, w - 1, h - 1])
    if b[2] < b[0] + 1: b[0], b[2] = sorted((b[0], b[2])); b[2] = min(w - 1, b[0] + 1)
    if b[3] < b[1] + 1: b[1], b[3] = sorted((b[1], b[3])); b[3] = min(h - 1, b[1] + 1)
    return b.astype(np.float32)

def seed_of(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--decoder_only_ckpt", default=f"{CKPT}/sam_decoder_finetune_best.pth")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.10, 0.20])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--invert", action="store_true")
    a = ap.parse_args()
    global OUT
    if a.invert:
        OUT = OUT.replace(".csv", "_inverted.csv")
    pairs = find_pairs(os.path.expanduser(a.root))
    items = []
    for img, masks in sorted(pairs.items()):
        g = np.zeros(cv2.imread(img, 0).shape, np.uint8)
        les = []
        for m in sorted(masks):
            mk = (cv2.imread(m, 0) > 127).astype(np.uint8)
            if mk.shape != g.shape:
                mk = cv2.resize(mk, g.shape[::-1], interpolation=cv2.INTER_NEAREST)
            if mk.sum():
                les.append(mk); g |= mk
        if len(les) == 1:
            n, lab = cv2.connectedComponents(les[0])
            comps = [c for c in ((lab == k).astype(np.uint8) for k in range(1, n)) if c.sum() >= 50]
            if len(comps) > 1:
                les = comps
        if g.sum():
            items.append((img, g, les))
    print(f"image/mask pairs: {len(pairs)} | with lesion: {len(items)} | classes:",
          pd.Series([cls_of(i[0]) for i in items]).value_counts().to_dict())
    dup = flag_duplicates([i[0] for i in items])
    print("near-duplicates flagged:", int(dup.sum()), "| images with >1 lesion:", sum(len(i[2]) > 1 for i in items))
    if a.test:
        items, dup = items[:15], dup[:15]
    predictor = SamPredictor(sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT).to(DEVICE).eval())
    ft = {"decoder_only": load_model("decoder_only", a.decoder_only_ckpt),
          "lsam_full": load_model("lsam_full", f"{CKPT}/lsam_full_best.pth")}
    enc = ft["lsam_full"][0]
    rows = []
    for k, ((img, gt, les), d) in enumerate(zip(items, dup)):
        gray = cv2.imread(img, 0)
        if a.invert:
            gray = 255 - gray
        h, w = gray.shape
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        s = max(h, w)
        sq = np.zeros((s, s, 3), np.uint8); sq[:h, :w] = rgb
        with torch.no_grad():
            predictor.set_image(rgb)
            x = torch.tensor(cv2.resize(sq, (IMG_SIZE, IMG_SIZE))).permute(2, 0, 1)[None].float().to(DEVICE)
            emb_raw = enc.image_encoder(enc.preprocess(x))
        prompts = [("merged", [tight_box(gt)])]
        if len(les) > 1:
            prompts.append(("per_lesion", [tight_box(m) for m in les]))
        name = os.path.relpath(img, os.path.expanduser(a.root))
        for level in a.levels:
            for rep in ([0] if level == 0 else range(a.reps)):
                for pname, oboxes in prompts:
                    boxes = [jitter(b, level, seed_of(name, level, rep, pname, i), w, h) for i, b in enumerate(oboxes)]
                    for model in ("sam_zeroshot", "decoder_only", "lsam_full"):
                        pred = np.zeros_like(gt)
                        for b in boxes:
                            if model == "sam_zeroshot":
                                ms, sc, _ = predictor.predict(box=b, multimask_output=True)
                                pred |= ms[int(np.argmax(sc))].astype(np.uint8)
                            else:
                                pred |= finetuned_mask(*ft[model], emb_raw, b, s, s)[:h, :w]
                        dsc, h_, h95, asd_, e = metrics(pred, gt)
                        rows.append(dict(file=name, cls=cls_of(img), height=h, width=w, n_lesions=len(les), dup=int(d),
                                         model=model, prompt=pname, level=level, rep=rep,
                                         DSC=dsc, HD=h_, HD95=h95, ASD=asd_, empty_pred=e))
        if (k + 1) % 50 == 0 or a.test:
            print(f"{k + 1}/{len(items)} images", flush=True)
            pd.DataFrame(rows).to_csv(OUT.replace(".csv", "_test.csv") if a.test else OUT, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT.replace(".csv", "_test.csv") if a.test else OUT, index=False)
    print(df[(df.level == 0) & (df.prompt == "merged")].groupby("model")[["DSC", "HD95", "ASD"]].mean().round(4))

if __name__ == "__main__":
    main()
