import os, argparse
import numpy as np, pandas as pd, cv2, torch
import torch.nn.functional as F
from segment_anything import sam_model_registry
from revision_eval import PROJECT_DIR, DEVICE
from box_jitter import tight_box
from busi_eval import find_pairs, cls_of, metrics, jitter, seed_of

OUT = f"{PROJECT_DIR}/lsam_project/results_revision/busi_medsam_per_image.csv"

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

@torch.no_grad()
def medsam_mask(model, emb, box, h, w):
    b = torch.tensor(box / np.array([w, h, w, h], np.float32) * 1024, dtype=torch.float32, device=DEVICE)[None, None]
    sp, de = model.prompt_encoder(points=None, boxes=b, masks=None)
    low, _ = model.mask_decoder(image_embeddings=emb, image_pe=model.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sp, dense_prompt_embeddings=de, multimask_output=False)
    p = F.interpolate(torch.sigmoid(low), size=(h, w), mode="bilinear", align_corners=False)
    return (p.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.10, 0.20])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    root = os.path.expanduser(a.root)
    items = load_items(root)
    print("lesion images:", len(items), flush=True)
    if a.test:
        items = items[:15]
    model = sam_model_registry["vit_b"](checkpoint=a.ckpt).to(DEVICE).eval()
    rows = []
    for k, (img, gt, les) in enumerate(items):
        rgb = cv2.cvtColor(cv2.imread(img, 0), cv2.COLOR_GRAY2RGB)
        h, w = gt.shape
        x = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        x = (x - x.min()) / np.clip(x.max() - x.min(), 1e-8, None)
        with torch.no_grad():
            emb = model.image_encoder(torch.tensor(x).permute(2, 0, 1)[None].to(DEVICE))
        prompts = [("merged", [tight_box(gt)])]
        if len(les) > 1:
            prompts.append(("per_lesion", [tight_box(m) for m in les]))
        name = os.path.relpath(img, root)
        for level in a.levels:
            for rep in ([0] if level == 0 else range(a.reps)):
                for pname, oboxes in prompts:
                    boxes = [jitter(b, level, seed_of(name, level, rep, pname, i), w, h) for i, b in enumerate(oboxes)]
                    pred = np.zeros_like(gt)
                    for b in boxes:
                        pred |= medsam_mask(model, emb, b, h, w)
                    dsc, h_, h95, asd_, e = metrics(pred, gt)
                    rows.append(dict(file=name, cls=cls_of(img), height=h, width=w, n_lesions=len(les), dup=0,
                                     model="medsam", prompt=pname, level=level, rep=rep,
                                     DSC=dsc, HD=h_, HD95=h95, ASD=asd_, empty_pred=e))
        if (k + 1) % 50 == 0 or a.test:
            print(f"{k + 1}/{len(items)} images", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT.replace(".csv", "_test.csv") if a.test else OUT, index=False)
    print(df[(df.level == 0) & (df.prompt == "merged")][["DSC", "HD95", "ASD"]].mean().round(4).to_string())

if __name__ == "__main__":
    main()
