# From SAM to LSAM: Prompt Formulation and Adaptation for Lesion Segmentation under Controlled Lesion Ambiguity

Code accompanying the manuscript submitted to the *IEEE Open Journal of Engineering in Medicine and Biology* (OJEMB-00183-2026).

Trained weights and all generated datasets are archived on Zenodo: **DOI: [DOI]**.

## Contents

| Folder | Purpose |
|---|---|
| `generators/` | Synthetic data: V2 generator (`generate_v2.py`, one seed per realization; seeds in `run_10_sam_experiments.sh`), earlier V2 generator used for the fixed artifact set (`generate_phantom_shapes_v2.py`, seed 42), artifact perturbations (`create_artifact_datasets.py`) |
| `sam_baseline/` | Slice conversion, zero-shot SAM inference, artifact evaluation, BUSI subset preparation |
| `lsam/` | Decoder-only fine-tuning and LSAM training, all revision evaluations (per-lesion boxes, box perturbation, OOD, BUSI with MedSAM and SAM 2.1, factorial design), statistics (Wilcoxon/Holm, bootstrap CIs, mixed-effects models) |
| `nnunet/` | nnU-Net 2D dataset setup and prediction (nnU-Net 3D: standard `nnUNetv2_train <ID> 3d_fullres 0`) |
| `figures/` | Scripts for the figures in the paper |

## Seeds

| Data | Script | Seed(s) |
|---|---|---|
| Ten V2 test realizations (run 1–10) | `generators/generate_v2.py` | 42, 123, 2024, 2025, 7, 77, 99, 1001, 1337, 9999 |
| Training set for decoder-only, LSAM and nnU-Net 2D | `generators/generate_v2.py` | 55555 |
| Fixed artifact set (earlier V2 generator) | `generators/generate_phantom_shapes_v2.py` | 42 |
| OOD sets (shape, appearance, combined) | `lsam/ood_generate.py` | set in the script (9000 + j, run offsets) |
| Factorial cells A–C | `lsam/r27_factorial.py` | set in the script (run offsets) |
| Box perturbations | `lsam/box_jitter.py` | fixed per slice (`seed_of`) |

LSAM and decoder-only training did not fix a training seed (see the paper). `generators/run_10_sam_experiments.sh` documents the test seeds and the original zero-shot SAM pipeline; its evaluation step (`evaluate_sam_full_metrics.py`) is superseded. All reported metrics are computed with `lsam/revision_eval.py`.

To regenerate the training set: `python generators/generate_v2.py --seed 55555 --out_dir LSAM_train_PhantomShapes_V2`

## Setup

```bash
conda create -n samlsam python=3.13 && conda activate samlsam
pip install -r requirements.txt
```

Download the SAM ViT-B checkpoint (`sam_vit_b_01ec64.pth`) from the official [segment-anything](https://github.com/facebookresearch/segment-anything) repository, MedSAM from [MedSAM](https://github.com/bowang-lab/MedSAM), and SAM 2.1 Hiera-B+ from [SAM 2](https://github.com/facebookresearch/sam2).

The scripts use absolute paths from the original machine. Point them to your own copy with:

```bash
grep -rl "/home/shqipe/Desktop" . | xargs sed -i "s#/home/shqipe/Desktop#$HOME/your/path#g"
```

## Reproducing the paper

1. **Data.** Either download the generated datasets from Zenodo or regenerate them:
   `python generators/generate_v2.py --seed <seed>` for each V2 realization (seeds in `run_10_sam_experiments.sh`), `python lsam/ood_generate.py` for the three OOD sets (seeds 9000+), `python lsam/r27_factorial.py` for the structure–appearance factorial cells.
2. **Training.** `python lsam/train_sam_decoder_finetune.py`, `python lsam/train_lsam_full.py`, `python lsam/train_lsam_ablation.py` (20 epochs, AdamW, lr 1e-5, weight decay 1e-4, λ = 0.3; no fixed training seed, see the paper). nnU-Net: `nnunet/setup_nnunet2d_dataset.sh`, then 100 epochs (`nnUNetTrainer_100epochs`), fold 0, final checkpoint.
3. **Evaluation.** `lsam/revision_eval.py` (all models on V2, tight boxes), `per_lesion_boxes.py`, `box_jitter.py`, `box_jitter_ablation.py`, `ood_eval.py`, `ood_eval_nnunet.py`, `busi_eval.py`, `busi_medsam.py`, `busi_sam2.py`, `nnunet2d_medpy.py`, `nnunet3d_recheck.py`; artifacts: `sam_baseline/artifact_case_level.py`.
4. **Statistics.** `lsam_comparison.py`, `analyze_per_lesion.py`, `analyze_jitter.py`, `analyze_jitter_ablation.py`, `analyze_ablation.py`, `factor_mixed_model.py`, `cnr_analysis.py`, `r27_analysis.py`, `ood_analysis.py`, `busi_analysis.py`, `sam_baseline/artifact_case_level_stats.py`.

All metrics use MedPy (`dc`, `hd`, `hd95`, `assd`); empty predictions are scored as DSC = 0 and distance = image diagonal.

## Real images

The curated BUSI dataset is not redistributed here. Obtain it from its original source (Zenodo DOI 10.5281/zenodo.19047974) and prepare it with `sam_baseline/prepare_busi_subset.py`.

## Citation

If you use this code, please cite the paper (citation will be added upon publication) and the Zenodo archive.

## License

Code: MIT License (see `LICENSE`). Generated data and weights on Zenodo: CC BY 4.0.
