# Corrections Before Final M.Tech Submission

## Required identity and contribution edits

- Replace the placeholder student author in `pyproject.toml`, `CITATION.cff`, and `LICENSE`.
- Use one exact project title in the report, slides, repository, and demonstration.
- State that the software is an independent educational implementation inspired by DeFusion.
- Add a student-contribution subsection covering the order-invariant architecture, Streamlit app,
  baseline implementations, evaluation tooling, tests, and any experiments actually completed.
- Cite the ECCV 2022 paper and distinguish it from this implementation.

## Required methodology edits

- Use **ResNeSt** only where the original projection-head design is being described. This repository
  uses custom residual blocks and must not be described as containing a ResNeSt backbone.
- Explain mask construction, Gaussian-noise corruption, common/unique targets, MAE terms, and the
  union-coverage constraint.
- Distinguish label-free/self-supervised training from training-free inference.
- State whether weights were trained by the student, for how long, and on which exact images.
- Describe image registration, color/channel conversion, resizing, normalization, and padding.

## Required results edits

- Remove or mark `SSIM = 0.831` as an unverified document claim until reproduced.
- Never call `source_ssim_proxy` ground-truth SSIM.
- Remove or mark `60 FPS` as a target until a controlled benchmark is saved.
- Separate results by fusion task and dataset.
- Report pair counts, means, standard deviations, hardware, precision, and checkpoint identifiers.
- Ensure the PDF and PPT contain identical measured numbers and baseline names.

## Required evidence

- Commit hash of the submitted code.
- Resolved training configuration and loss-history JSONL.
- Final checkpoint hash and file location.
- Per-pair metrics CSV and summary JSON.
- Original/fused qualitative examples with captions and dataset attribution.
- Test output and a short demonstration procedure.

## Recommended final demonstration

1. Show a classical baseline so the app works without a model checkpoint.
2. Load the trained CUD checkpoint.
3. Fuse one pair from each application type.
4. Display common and unique projections.
5. Explain the metric definitions and limitations.
6. Open the saved experiment directory containing the configuration and per-pair evidence.
