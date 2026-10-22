import os, re, glob, zlib, argparse
from collections import defaultdict
import numpy as np, pandas as pd, cv2, torch
from medpy.metric.binary import dc, hd, hd95, assd
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

OUT = os.path.expanduser("~/Desktop/sam_phantom_project/lsam_project/results_revision/busi_sam2_per_image.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def tight_box(mask):
    ys, xs = np.where(mask > 0)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

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

def load_items(root):
    items = []
    for img, masks in sorted(find_pairs(root).items()):
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
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.10, 0.20])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    root = os.path.expanduser(a.root)
    items = load_items(root)
    print("lesion images:", len(items), flush=True)
    if a.test:
        items = items[:15]
    pred = SAM2ImagePredictor(build_sam2(a.cfg, a.ckpt, device=DEVICE))
    rows = []
    for k, (img, gt, les) in enumerate(items):
        rgb = cv2.cvtColor(cv2.imread(img, 0), cv2.COLOR_GRAY2RGB)
        h, w = gt.shape
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            pred.set_image(rgb)
            prompts = [("merged", [tight_box(gt)])]
            if len(les) > 1:
                prompts.append(("per_lesion", [tight_box(m) for m in les]))
            name = os.path.relpath(img, root)
            for level in a.levels:
                for rep in ([0] if level == 0 else range(a.reps)):
                    for pname, oboxes in prompts:
                        boxes = [jitter(b, level, seed_of(name, level, rep, pname, i), w, h) for i, b in enumerate(oboxes)]
                        p = np.zeros_like(gt)
                        for b in boxes:
                            ms, sc, _ = pred.predict(box=b, multimask_output=True)
                            p |= (ms[int(np.argmax(sc))] > 0).astype(np.uint8)
                        dsc, h_, h95, asd_, e = metrics(p, gt)
                        rows.append(dict(file=name, cls=cls_of(img), height=h, width=w, n_lesions=len(les), dup=0,
                                         model="sam2", prompt=pname, level=level, rep=rep,
                                         DSC=dsc, HD=h_, HD95=h95, ASD=asd_, empty_pred=e))
        if (k + 1) % 50 == 0 or a.test:
            print(f"{k + 1}/{len(items)} images", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT.replace(".csv", "_test.csv") if a.test else OUT, index=False)
    print(df[(df.level == 0) & (df.prompt == "merged")][["DSC", "HD95", "ASD"]].mean().round(4).to_string())

if __name__ == "__main__":
    main()
