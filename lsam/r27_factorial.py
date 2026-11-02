import os, argparse
import numpy as np, pandas as pd, cv2
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.draw import disk, polygon
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, label as cc_label

IMG_DIM, N_SLICES = 256, 5
OUT = "results_revision"

def create_base_phantom():
    p = resize(shepp_logan_phantom(), (IMG_DIM, IMG_DIM), anti_aliasing=True)
    return ((p * 2000) - 1000).astype(np.float32)

def random_center():
    return np.random.randint(60, 196), np.random.randint(60, 196)

def create_shape_mask(name):
    mask = np.zeros((IMG_DIM, IMG_DIM), dtype=np.uint8)
    cy, cx = random_center()
    if name == "circle":
        rr, cc = disk((cy, cx), np.random.randint(12, 28), shape=mask.shape); mask[rr, cc] = 1
    elif name == "square":
        s = np.random.randint(25, 45)
        mask[max(0, cy - s // 2):min(IMG_DIM, cy + s // 2), max(0, cx - s // 2):min(IMG_DIM, cx + s // 2)] = 1
    else:
        s = np.random.randint(30, 55)
        pts = np.array([[cy - s // 2, cx], [cy + s // 2, cx - s // 2], [cy + s // 2, cx + s // 2]])
        rr, cc = polygon(pts[:, 0], pts[:, 1], shape=mask.shape); mask[rr, cc] = 1
    return mask

def make_irregular(mask):
    f = gaussian_filter(np.random.rand(IMG_DIM, IMG_DIM), sigma=np.random.uniform(3, 8))
    f = f > np.percentile(f, 55)
    irr = mask.copy().astype(bool)
    irr = binary_dilation(irr, iterations=np.random.randint(1, 3)) if np.random.rand() > 0.5 else binary_erosion(irr, iterations=1)
    return np.logical_and(irr, f | mask.astype(bool)).astype(np.uint8)

def move_to_edge(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask
    sy, sx = np.random.choice([-50, -35, 35, 50]), np.random.choice([-50, -35, 35, 50])
    new = np.zeros_like(mask); new[np.clip(ys + sy, 0, IMG_DIM - 1), np.clip(xs + sx, 0, IMG_DIM - 1)] = 1
    return new

def add_noise_and_blur(img, noise_sigma, blur_sigma, scale):
    img = img * scale + np.random.normal(0, noise_sigma, img.shape)
    if blur_sigma > 0:
        img = gaussian_filter(img, sigma=blur_sigma)
    return np.clip(img, -1000, 1000).astype(np.float32)

def appearance(mode):
    if mode == "v1":
        return 10, 0.5, float(np.random.uniform(0.9, 1.1))
    return int(np.random.choice([8, 12, 18, 25, 35])), float(np.random.choice([0.3, 0.5, 0.8, 1.1])), float(np.random.uniform(0.85, 1.15))

_AUX = np.random.RandomState(0)

def generate_case(base, empty, structure, app, design="revised", aux=None):
    """design="original": the structure switch toggles all four lesion-level factors (incl. low contrast).
    design="revised": structure toggles lesion geometry only (irregular, multiple, peripheral displacement);
    low contrast belongs to the appearance switch, together with noise, blur and intensity scaling.
    The global random stream is consumed exactly as in the original design, so cell A is generated
    identically and revised B/C keep the original masks, classes and nuisance parameters; when the contrast
    flag changes, the lesion intensity is redrawn from the auxiliary generator `aux`."""
    aux = _AUX if aux is None else aux
    img_vol = np.zeros((IMG_DIM, IMG_DIM, N_SLICES), np.float32); lbl_vol = np.zeros_like(img_vol, dtype=np.uint8)
    if empty:
        for z in range(N_SLICES):
            img_vol[:, :, z] = add_noise_and_blur(base, *appearance(app))
        return img_vol, lbl_vol, dict(cls="empty", variations="empty", noise=0, blur=0, scale=1.0)
    cls = np.random.choice(["circle", "square", "triangle"])
    irr, mult, low, edge = (np.random.rand() < 0.65, np.random.rand() < 0.35, np.random.rand() < 0.45, np.random.rand() < 0.35)
    low_original = low and structure            # contrast flag under the original switch definition
    if not structure:
        irr = mult = edge = False
    if design == "original":
        low = low_original
    elif app != "v2":
        low = False
    noise, blur, scale = appearance(app)
    var = [n for n, f in (("irregular", irr), ("multiple_lesions", mult), ("low_contrast", low), ("partial_edge", edge)) if f]
    final = np.zeros((IMG_DIM, IMG_DIM), np.uint8)
    for _ in range(np.random.randint(2, 4) if mult else 1):
        m = create_shape_mask(cls)
        if irr: m = make_irregular(m)
        if edge: m = move_to_edge(m)
        final = np.maximum(final, m)
    inten = np.random.randint(80, 250) if low_original else np.random.randint(350, 600)   # original draw
    if low != low_original:
        inten = aux.randint(80, 250) if low else aux.randint(350, 600)
    for z in range(N_SLICES):
        sm = final.copy()
        if z in (0, N_SLICES - 1) and np.random.rand() < 0.5:
            sm = binary_erosion(sm, iterations=1).astype(np.uint8)
        img = base.copy(); img[sm > 0] = inten
        img_vol[:, :, z] = add_noise_and_blur(img, noise, blur, scale); lbl_vol[:, :, z] = (sm > 0)
    return img_vol, lbl_vol, dict(cls=cls, variations=";".join(var) or "standard", noise=noise, blur=blur, scale=scale)

def to_uint8(x):
    return ((np.clip(x.astype(np.float32), -1000, 1000) + 1000) / 2000.0 * 255).astype(np.uint8)

def orient(a):
    return np.flipud(np.rot90(a, k=1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--design", choices=["revised", "original"], default="revised",
                    help="revised: low contrast in the appearance switch (cells B and C change, A and D do not)")
    ap.add_argument("--cells", default=None, help="comma-separated subset of A,B,C (default: B,C revised; A,B,C original)")
    a = ap.parse_args()
    cell_list = (a.cells or ("B,C" if a.design == "revised" else "A,B,C")).split(",")
    tag = "r27" if a.design == "original" else "r27_revised"
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    from revision_eval import compute_metrics, SAM_CHECKPOINT, DEVICE
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT).to(DEVICE).eval()
    pred = SamPredictor(sam)
    def best(box):
        m, s, _ = pred.predict(box=box, multimask_output=True); return m[int(np.argmax(s))].astype(np.uint8)
    def tbox(m):
        ys, xs = np.where(m > 0); return np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)
    # same seed offsets as the original design: realizations stay independent across cells, and each revised
    # B/C realization is paired with its original counterpart (identical masks and nuisance parameters; only
    # lesion contrast differs), which also gives a paired estimate of the low-contrast effect
    cells = {k: v for k, v in {"A": (False, "v1", 7000), "B": (True, "v1", 7100), "C": (False, "v2", 7200)}.items()
             if k in cell_list}
    nrun, nvol = (1, 40) if a.test else (10, 150)
    base = create_base_phantom()
    rows = []
    os.makedirs(OUT, exist_ok=True)
    for cell, (structure, app, s0) in cells.items():
        for r in range(1, nrun + 1):
            np.random.seed(s0 + r)
            aux = np.random.RandomState(100000 + s0 + r)
            for i in range(1, nvol + 1):
                img, lbl, meta = generate_case(base, i <= 30, structure, app, a.design, aux)
                if i <= 30:
                    continue
                for z in range(N_SLICES):
                    gt = (orient(lbl[:, :, z]) > 0).astype(np.uint8)
                    if gt.sum() == 0:
                        continue
                    rgb = cv2.cvtColor(np.ascontiguousarray(orient(to_uint8(img[:, :, z]))), cv2.COLOR_GRAY2RGB)
                    with torch.no_grad():
                        pred.set_image(rgb)
                        merged = best(tbox(gt))
                        lab, n = cc_label(gt)
                        per = merged if n <= 1 else np.max([best(tbox((lab == k).astype(np.uint8))) for k in range(1, n + 1)], axis=0)
                    for prompt, p in (("merged", merged), ("per_lesion", per)):
                        d, h, h95, s, e = compute_metrics(p, gt)
                        rows.append(dict(cell=cell, design=a.design, structure=int(structure), appearance=app, run=f"{cell}{r}",
                                         case_id=f"Case{i:03d}", slice=z, prompt=prompt, n_components=n,
                                         DSC=d, HD=h, HD95=h95, ASD=s, empty_pred=e, **meta))
            pd.DataFrame(rows).to_csv(f"{OUT}/{tag}_per_slice{'_test' if a.test else ''}.csv", index=False)
            print(f"cell {cell} run {r} done", flush=True)
    df = pd.DataFrame(rows)
    print(df[df.prompt == "merged"].groupby(["cell", "run"])[["DSC", "HD95"]].mean().groupby("cell").mean().round(4))

if __name__ == "__main__":
    main()
