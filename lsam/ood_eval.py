import os, argparse
import numpy as np, pandas as pd, nibabel as nib, cv2, torch
from scipy import ndimage
from segment_anything import sam_model_registry, SamPredictor
from revision_eval import load_model, compute_metrics, PROJECT_DIR, SAM_CHECKPOINT, IMG_SIZE, DEVICE
from box_jitter import finetuned_mask, tight_box
from r27_factorial import to_uint8, orient

OOD = f"{PROJECT_DIR}/OOD_datasets"
OUT = f"{PROJECT_DIR}/lsam_project/results_revision/ood_per_slice.csv"
CKPT = f"{PROJECT_DIR}/lsam_project/checkpoints"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder_only_ckpt", default=f"{CKPT}/sam_decoder_finetune_best.pth")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    predictor = SamPredictor(sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT).to(DEVICE).eval())
    ft = {"decoder_only": load_model("decoder_only", a.decoder_only_ckpt),
          "lsam_full": load_model("lsam_full", f"{CKPT}/lsam_full_best.pth")}
    enc = ft["lsam_full"][0]
    jobs = [("shape", 1)] if a.test else [(c, r) for c in ("shape", "appearance", "both") for r in range(1, 11)]
    rows = []
    for cell, r in jobs:
        d = f"{OOD}/{cell}_run{r}"
        meta = pd.read_csv(f"{d}/metadata.csv"); meta = meta[meta.cls != "empty"]
        if a.test:
            meta = meta.head(20)
        for _, m in meta.iterrows():
            img = nib.load(f"{d}/imagesTs/{m.case_id}_0000.nii.gz").get_fdata()
            lbl = nib.load(f"{d}/labelsTs/{m.case_id}.nii.gz").get_fdata()
            multi = "multiple_lesions" in str(m.variations)
            for z in range(img.shape[2]):
                gt = (orient(lbl[:, :, z]) > 0).astype(np.uint8)
                if gt.sum() == 0:
                    continue
                rgb = cv2.cvtColor(np.ascontiguousarray(orient(to_uint8(img[:, :, z]))), cv2.COLOR_GRAY2RGB)
                h, w = gt.shape
                with torch.no_grad():
                    predictor.set_image(rgb)
                    x = torch.tensor(cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))).permute(2, 0, 1)[None].float().to(DEVICE)
                    emb_raw = enc.image_encoder(enc.preprocess(x))
                lab, n = ndimage.label(gt)
                prompts = [("merged", [tight_box(gt)])]
                if multi and n > 1:
                    prompts.append(("per_lesion", [tight_box(lab == k) for k in range(1, n + 1)]))
                for model in ("sam_zeroshot", "decoder_only", "lsam_full"):
                    for pname, boxes in prompts:
                        pred = np.zeros_like(gt)
                        for b in boxes:
                            if model == "sam_zeroshot":
                                ms, sc, _ = predictor.predict(box=b, multimask_output=True)
                                pred |= ms[int(np.argmax(sc))].astype(np.uint8)
                            else:
                                pred |= finetuned_mask(*ft[model], emb_raw, b, w, h)
                        dsc, hd, h95, asd, e = compute_metrics(pred, gt)
                        rows.append(dict(cell=cell, run_id=f"run{r}", case_id=m.case_id, slice=z, model=model,
                                         prompt=pname, multi=multi, n_components=n, variations=m.variations,
                                         DSC=dsc, HD=hd, HD95=h95, ASD=asd, empty_pred=e))
        df = pd.DataFrame(rows)
        df.to_csv(OUT.replace(".csv", "_test.csv") if a.test else OUT, index=False)
        s = df[(df.cell == cell) & (df.run_id == f"run{r}") & (df.prompt == "merged")]
        print(f"{cell} run{r} done | " + "  ".join(f"{k}: HD95 {g.HD95.mean():.3f} DSC {g.DSC.mean():.4f}"
                                                  for k, g in s.groupby("model")), flush=True)

if __name__ == "__main__":
    main()
