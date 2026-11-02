# Generation of the out-of-distribution (OOD) test sets

Implemented in `lsam/ood_generate.py`. The three OOD test sets (*shape*, *appearance*, *combined*) share with V2 the Shepp–Logan phantom background (256 × 256 px, HU-equivalent range [−1000, 1000]), the five-slice volume layout, the four lesion-level ambiguity factors and their activation probabilities (Table 1 of the paper), and the slice conversion, prompts and metrics. Each set has ten realizations of 150 volumes; volumes 1–30 of every realization are lesion-free.

## Lesion shape (shape and combined sets)

Circles, squares and triangles are replaced by smooth random closed curves. Per lesion:

- centre (c_y, c_x) uniform in [60, 195] px per axis; base radius r0 uniform in the integers 12–27 px; number of control points k uniform in the integers 5–9;
- control point j = 0, …, k−1 at angle θ_j = 2π (j + u_j) / k with u_j ~ U(−0.3, 0.3), and radius ρ_j = r0 · v_j with v_j ~ U(0.55, 1.45) (radii vary by up to ±45%);
- a periodic cubic B-spline through the control points (`scipy.interpolate.splprep`, s = 0, periodic) is evaluated at 400 points and the enclosed polygon is filled.

The ambiguity factors are then applied as in V2, in this order: irregular boundary (if active), peripheral displacement (if active), union over the 2–3 independently drawn lesions (if multiple lesions are active). The outer two slices are eroded by one pixel with probability 0.5 each.

## Image model (appearance and combined sets)

The additive Gaussian noise, blur and global scaling of V2 are replaced by the model below. All parameters and the three random fields are drawn once per volume and shared by its five slices; only the noise is drawn anew for every slice.

A normalized random field with correlation length σ is F_σ = (G_σ ∗ W) / max|G_σ ∗ W|, with W ~ N(0, 1) i.i.d. per pixel and G_σ a Gaussian kernel of standard deviation σ px, so F_σ ∈ [−1, 1].
Lesion texture T = F_σt, background texture U = F_σu, bias field B = 1 + b · F_40.

Each slice is rendered from the phantom P and the slice mask M in this order:

1. Lesion intensity and texture: I1 = I_L + a_t · T inside M, P elsewhere
2. Background texture (whole image): I2 = I1 + a_u · U
3. Bias field and global scaling: I3 = (I2 + 1000) · B · s − 1000
4. Signal-dependent noise: I4 = I3 + k · sqrt(max(I3 + 1000, 0) + 1) · ε, ε ~ N(0, 1) per pixel
5. Blur: I5 = G_σb ∗ I4
6. Clip to [−1000, 1000]

The lesion intensity I_L is drawn once per volume from the V2 ranges (integers 80–249 HU if low contrast is active, otherwise 350–599 HU). Lesion-free volumes are rendered with the same steps without a lesion. The shape set keeps the V2 image model.

| Parameter (drawn once per volume) | Symbol | Distribution |
|---|---|---|
| Lesion texture amplitude (HU) | a_t | U(30, 90) |
| Lesion texture correlation length (px) | σ_t | U(1.5, 4) |
| Background texture amplitude (HU) | a_u | U(15, 40) |
| Background texture correlation length (px) | σ_u | U(3, 8) |
| Bias-field strength | b | U(0.08, 0.20) |
| Bias-field correlation length (px) | – | 40 (fixed) |
| Noise coefficient | k | U(0.3, 1.1) |
| Blur (px) | σ_b | uniform on {0.3, 0.5, 0.8, 1.1} |
| Global intensity scaling | s | U(0.85, 1.15) |

## Seeds

Realization r = 1, …, 10 of the shape, appearance and combined sets uses the NumPy global seed 8100 + r, 8200 + r and 8300 + r; all draws follow from this seed in the order of the script. The example figure uses seeds 9000 + j, j = 0, …, 3.
