# Evaluation Protocol

## 1. Freeze the experiment before running it

Record the commit hash, checkpoint hash, dataset version, pair list, image preprocessing, random
seed, hardware, Python/PyTorch versions, precision, and command used. Do not update the report table
from an informal Streamlit run.

## 2. Evaluate tasks separately

Create independent result tables for:

- multi-exposure fusion,
- multi-focus fusion,
- infrared-visible fusion.

Do not average all tasks into a single unexplained number. Every method must receive the same
registered pairs and preprocessing.

## 3. Use appropriate datasets

The original DeFusion work reports experiments on public datasets such as MEFB/SICE for
multi-exposure, Real-MFF/MFIFB for multi-focus, and RoadScene/TNO for infrared-visible fusion.
Verify each dataset's current license and citation before downloading or redistributing it.

## 4. Interpret the included metrics correctly

- `entropy_bits` measures fused-image intensity information, not semantic usefulness.
- `spatial_frequency` measures activity/sharpness and can also reward noise.
- `mutual_information_sum` measures dependence between the fused result and both sources.
- `source_ssim_proxy` averages SSIM from the fused result to each source. It is not SSIM against a
  ground-truth fused image.

Add accepted task-specific metrics such as MEF-SSIM, VIF, QAB/F, SCD, or ground-truth metrics only
when their implementation and reference definitions are documented. Never relabel the included
proxy as ordinary SSIM.

## 5. Report uncertainty and sample count

For every metric, report the number of image pairs, mean, standard deviation, and preferably a
paired statistical comparison. Include per-pair CSV data as an experiment artifact.

## 6. Benchmark runtime honestly

State device, image resolution, batch size, precision, warm-up count, timed repetitions, and whether
loading/preprocessing/UI time is included. Report median and tail latency, not just the fastest run.
Convert latency to FPS only for a clearly defined steady-state benchmark. The Streamlit timer is a
single-run diagnostic and is not evidence for a 60 FPS claim.

## 7. Qualitative review

Show the same representative pairs for all methods. Use crops to inspect focus boundaries,
exposure recovery, infrared targets, color distortion, halos, and noise. Keep original images and
uncompressed fused outputs available for examination.

## 8. Reproducible table template

| Task | Dataset | Pairs | Method | Metric | Mean | Std. dev. | Commit | Checkpoint |
|---|---|---:|---|---|---:|---:|---|---|
| Multi-focus | _fill_ | _fill_ | CUDNet | _fill_ | _measured_ | _measured_ | _hash_ | _hash_ |

Values such as `0.831` or `60 FPS` belong in the final report only after this table can be completed
with saved evidence.

## 9. Repository evaluator

The matched-folder evaluator supports the five classical baselines and a trained CUD checkpoint.
When CUD is selected, `--checkpoint` is mandatory. Supply `--task`, `--dataset-name`, and `--commit`
so `summary.json` records the experiment context. The output includes per-pair CSV measurements,
sample means and standard deviations, runtime, environment information, and the checkpoint SHA-256.

The evaluator does not automatically make an experiment publication-ready. Preserve the dataset
pair list and license information separately, verify metric implementations against their cited
definitions, and repeat controlled runtime measurements with warm-up before reporting FPS.
