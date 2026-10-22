import os, glob, argparse
import numpy as np, pandas as pd, nibabel as nib
from scipy.ndimage import gaussian_filter, binary_erosion
from scipy.interpolate import splprep, splev
from skimage.draw import polygon
from r27_factorial import (create_base_phantom, random_center, create_shape_mask, make_irregular,
                           move_to_edge, add_noise_and_blur, appearance, IMG_DIM, N_SLICES)

PROJECT = os.path.expanduser("~/Desktop/sam_phantom_project")
OUT = f"{PROJECT}/OOD_datasets"
REF = sorted(glob.glob(f"{PROJECT}/SAM_10_runs/PhantomShapes_V2_run1/imagesTr/*_0000.nii.gz"))
CELLS = {"shape": (True, False, 8100), "appearance": (False, True, 8200), "both": (True, True, 8300)}

def spline_mask():
    mask = np.zeros((IMG_DIM, IMG_DIM), np.uint8)
    cy, cx = random_center()
    r0, k = np.random.randint(12, 28), np.random.randint(5, 10)
    ang = (np.arange(k) + np.random.uniform(-0.3, 0.3, k)) * 2 * np.pi / k
    rad = r0 * np.random.uniform(0.55, 1.45, k)
    x, y = cx + rad * np.cos(ang), cy + rad * np.sin(ang)
    tck, _ = splprep([np.r_[x, x[0]], np.r_[y, y[0]]], s=0, per=True)
    xs, ys = splev(np.linspace(0, 1, 400), tck)
    rr, cc = polygon(ys, xs, shape=mask.shape); mask[rr, cc] = 1
    return mask

def unit_field(sigma):
    f = gaussian_filter(np.random.randn(IMG_DIM, IMG_DIM), sigma)
    return f / (np.abs(f).max() + 1e-8)

def ood_appearance_params():
    return dict(tex_amp=float(np.random.uniform(30, 90)), tex_sigma=float(np.random.uniform(1.5, 4)),
                bg_amp=float(np.random.uniform(15, 40)), bg_sigma=float(np.random.uniform(3, 8)),
                bias=float(np.random.uniform(0.08, 0.20)), noise_k=float(np.random.uniform(0.3, 1.1)),
                blur=float(np.random.choice([0.3, 0.5, 0.8, 1.1])), scale=float(np.random.uniform(0.85, 1.15)))

def render_ood(base, mask, inten, p, tex, bg, field):
    img = base.copy()
    if mask is not None:
        img[mask > 0] = inten + p["tex_amp"] * tex[mask > 0]
    img = img + p["bg_amp"] * bg
    img = (img + 1000) * field * p["scale"] - 1000
    img = img + np.random.normal(0, 1, img.shape) * p["noise_k"] * np.sqrt(np.clip(img + 1000, 0, None) + 1)
    img = gaussian_filter(img, p["blur"])
    return np.clip(img, -1000, 1000).astype(np.float32)

def generate_case(base, empty, new_shape, new_app):
    img_vol = np.zeros((IMG_DIM, IMG_DIM, N_SLICES), np.float32)
    lbl_vol = np.zeros_like(img_vol, dtype=np.uint8)
    if new_app:
        p = ood_appearance_params()
        tex, bg, field = unit_field(p["tex_sigma"]), unit_field(p["bg_sigma"]), 1 + p["bias"] * unit_field(40)
        app = dict(appearance="ood", **p)
    else:
        noise, blur, scale = appearance("v2")
        app = dict(appearance="v2", noise=noise, blur=blur, scale=scale)
    if empty:
        for z in range(N_SLICES):
            img_vol[:, :, z] = (render_ood(base, None, 0, p, tex, bg, field) if new_app
                                else add_noise_and_blur(base, *appearance("v2")))
        return img_vol, lbl_vol, dict(cls="empty", variations="empty", **app)
    cls = "spline" if new_shape else str(np.random.choice(["circle", "square", "triangle"]))
    irr, mult, low, edge = (np.random.rand() < 0.65, np.random.rand() < 0.35,
                            np.random.rand() < 0.45, np.random.rand() < 0.35)
    var = [n for n, f in (("irregular", irr), ("multiple_lesions", mult), ("low_contrast", low),
                          ("partial_edge", edge)) if f]
    final = np.zeros((IMG_DIM, IMG_DIM), np.uint8)
    for _ in range(np.random.randint(2, 4) if mult else 1):
        m = spline_mask() if new_shape else create_shape_mask(cls)
        if irr: m = make_irregular(m)
        if edge: m = move_to_edge(m)
        final = np.maximum(final, m)
    inten = np.random.randint(80, 250) if low else np.random.randint(350, 600)
    for z in range(N_SLICES):
        sm = final.copy()
        if z in (0, N_SLICES - 1) and np.random.rand() < 0.5:
            sm = binary_erosion(sm, iterations=1).astype(np.uint8)
        if new_app:
            img_vol[:, :, z] = render_ood(base, sm, inten, p, tex, bg, field)
        else:
            img = base.copy(); img[sm > 0] = inten
            img_vol[:, :, z] = add_noise_and_blur(img, app["noise"], app["blur"], app["scale"])
        lbl_vol[:, :, z] = sm > 0
    return img_vol, lbl_vol, dict(cls=cls, variations=";".join(var) or "standard", **app)

def example_figure(path, base):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from r27_factorial import to_uint8, orient
    fig, ax = plt.subplots(2, 4, figsize=(12, 6.4))
    for j, (name, (ns, na, _)) in enumerate([("V2 (in-distribution)", (False, False, 0))] +
                                            [(f"OOD-{k}", v) for k, v in CELLS.items()]):
        np.random.seed(9000 + j)
        for i in range(2):
            img, lbl, _ = generate_case(base, False, ns, na)
            ax[i, j].imshow(orient(to_uint8(img[:, :, 2])), cmap="gray", vmin=0, vmax=255)
            ax[i, j].contour(orient(lbl[:, :, 2]), [0.5], colors="r", linewidths=0.6)
            ax[i, j].axis("off")
        ax[0, j].set_title(name, fontsize=11)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); a = ap.parse_args()
    affine = nib.load(REF[0]).affine if REF else np.eye(4)
    print("reference:", REF[0] if REF else "none found, identity affine", "| affine diag:", np.diag(affine))
    nrun, nvol = (1, 40) if a.test else (10, 150)
    base = create_base_phantom()
    for cell, (new_shape, new_app, s0) in CELLS.items():
        for r in range(1, nrun + 1):
            np.random.seed(s0 + r)
            d = f"{OUT}/{cell}_run{r}" + ("_test" if a.test else "")
            os.makedirs(f"{d}/imagesTs", exist_ok=True); os.makedirs(f"{d}/labelsTs", exist_ok=True)
            rows = []
            for i in range(1, nvol + 1):
                img, lbl, meta = generate_case(base, i <= 30, new_shape, new_app)
                cid = f"Case{i:03d}"
                nib.save(nib.Nifti1Image(img, affine), f"{d}/imagesTs/{cid}_0000.nii.gz")
                nib.save(nib.Nifti1Image(lbl, affine), f"{d}/labelsTs/{cid}.nii.gz")
                rows.append(dict(case_id=cid, **meta))
            m = pd.DataFrame(rows); m.to_csv(f"{d}/metadata.csv", index=False)
            les = m[m.cls != "empty"]
            freq = {k: round(float(les.variations.str.contains(k).mean()), 2)
                    for k in ("irregular", "multiple", "low_contrast", "partial_edge")}
            print(f"{cell} run{r}: {len(les)} lesion volumes, classes {les.cls.unique().tolist()}, factors {freq}", flush=True)
    if a.test:
        os.makedirs("results_revision", exist_ok=True)
        example_figure("results_revision/ood_examples.png", base)
        if REF:
            ref = nib.load(REF[0]); new = nib.load(f"{OUT}/shape_run1_test/imagesTs/Case031_0000.nii.gz")
            print("shape/dtype V2:", ref.shape, ref.get_data_dtype(), "| OOD:", new.shape, new.get_data_dtype())
        print("example figure: results_revision/ood_examples.png")

if __name__ == "__main__":
    main()
