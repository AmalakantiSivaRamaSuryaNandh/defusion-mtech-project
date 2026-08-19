# Methodology

## Problem definition

Given two registered RGB images of the same scene, the system estimates one fused RGB image that
preserves useful common and complementary information. Registration is treated as a prerequisite;
this version does not estimate homographies, optical flow, or cross-modal correspondences.

## Self-supervised CUD task

For each unlabeled training image `x`, two binary masks `M1` and `M2` are sampled with the constraint
that their union covers every pixel. Independent Gaussian-noise tensors replace the masked-out
regions:

```text
x1 = M1 * x + (1 - M1) * n1
x2 = M2 * x + (1 - M2) * n2
```

The exact training targets are therefore available without human labels:

```text
common  = x * M1 * M2
unique1 = x * M1 * (1 - M2)
unique2 = x * (1 - M1) * M2
scene   = x
```

The loss is the sum of four mean-absolute-error terms: the three decomposition projections and the
full-scene reconstruction.

## Network

The compact `CUDNet` contains:

1. A shared three-stage residual encoder that reduces resolution by a factor of eight.
2. An ensembler that receives the mean and absolute difference of the two encoded views. This makes
   the common representation independent of input order.
3. A common decoder and a shared unique decoder.
4. Projection heads for common and unique images.
5. A reconstruction head that receives common features and the sum of the two unique features.

The summed unique representation makes the fused output invariant to swapping source A and B. This
is a deliberate implementation choice and should be reported as a modification from the broad
paper concept, not attributed to the original authors without evidence.

## Inference

At inference, the two aligned source images pass through the trained model without fine-tuning. The
reconstruction head generates the fused image, while the three decomposition projections can be
displayed for interpretation. Inputs are reflect-padded to multiples of eight and cropped back to
their original size.

## Baselines

- **Average:** pixel-wise arithmetic mean.
- **PCA:** global first-principal-component weights estimated from the two flattened inputs.
- **Laplacian pyramid:** selects high-energy detail coefficients and averages the coarsest level.
- **Wavelet:** averages approximation coefficients and selects detail coefficients by magnitude.
- **Local focus:** blends sources using smoothed local Laplacian energy.

Baselines provide a runnable comparison before the deep model is trained. They are not substitutes
for reproducing the published comparison against methods such as DenseFuse, IFCNN, or U2Fusion.

## Relationship to DeFusion

The project follows the CUD pretext-task idea from Liang et al. (ECCV 2022), including mask-and-noise
view generation, common/unique projections, MAE supervision, and scene reconstruction. It is not a
line-for-line reproduction of the official architecture. The official implementation remains the
appropriate reference for exact reproduction claims.
