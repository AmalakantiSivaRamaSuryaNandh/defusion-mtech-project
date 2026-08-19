# Label-Free Image Fusion Through Self-Supervised Feature Decomposition

This repository is a runnable M.Tech project for combining two aligned source images into a
single fused result. It includes five classical baselines, a compact PyTorch network trained by a
Common and Unique Decomposition (CUD) pretext task, a Streamlit interface, batch evaluation,
tests, and an explicit protocol for reporting results.

The deep model is an **independent educational reproduction inspired by DeFusion**, not the
authors' official implementation. It follows the method's central idea: create two mask-and-noise
views of one unlabeled image, learn their common and unique components, and reconstruct the clean
scene. See the [ECCV 2022 paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780706.pdf)
and [official DeFusion repository](https://github.com/erfect2020/DecompositionForFusion).

## What the project implements

- Average, PCA, Laplacian-pyramid, wavelet, and local-focus fusion.
- An order-invariant CUD-inspired encoder/ensembler/decoder architecture.
- Self-supervised targets for common, source-A unique, source-B unique, and reconstructed images.
- COCO-compatible unlabeled-image training with reproducible configuration and checkpoints.
- A Streamlit application with input validation, downloads, transparent metrics, and feature views.
- CLI inference, matched-folder evaluation, synthetic demo generation, unit tests, linting, and CI.

The software can support multi-focus, multi-exposure, and infrared-visible experiments, but the
source images must already be geometrically registered. Resizing two images to the same dimensions
is not a substitute for registration.

## Academic integrity and result status

No pretrained weights or experimental datasets are committed. Classical methods work immediately;
the CUD option requires a checkpoint produced by the training command below. The figures `SSIM =
0.831` and `60 FPS` mentioned in the project documents are **not reproduced by this repository**.
Only results generated with a named dataset, protocol, hardware configuration, and saved output
should be included in the final report.

The metric named `source_ssim_proxy` is the average SSIM between the fused image and the two source
images. It is not SSIM against a ground-truth fused image and must not be described that way.

## Installation

Python 3.10-3.13 is supported. A virtual environment is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For a CUDA-enabled PyTorch build, install the wheel recommended for your CUDA version from the
[official PyTorch setup page](https://pytorch.org/get-started/locally/) before installing this
project.

## Run the web application

```bash
streamlit run app.py
```

Upload two images, select a method, fuse them, review the metrics, and download the PNG result. For
the CUD method, enter a path to a locally trained checkpoint such as `runs/cud/final.pt`.

## Quick classical-method demo

```bash
python scripts/create_demo_pair.py
defusion-fuse \
  --image-a examples/demo_pair/source_a.png \
  --image-b examples/demo_pair/source_b.png \
  --method laplacian \
  --output runs/demo/fused.png \
  --metrics-json runs/demo/metrics.json
```

On Windows PowerShell, place the command on one line or use PowerShell's continuation syntax.

## Train the CUD model

Download or prepare an unlabeled natural-image folder. The original DeFusion paper used 50,000 COCO
images, 256x256 crops, Adam, 50 epochs, batch size 8, initial learning rate `1e-3`, and a factor-of-2
learning-rate reduction every ten epochs. The checked-in configuration mirrors those values.

```bash
defusion-train --config configs/train_cud.json --data-dir data/coco/train2017
```

For a small pipeline check before a long run:

```bash
defusion-train \
  --data-dir path/to/a/small/image/folder \
  --output-dir runs/smoke \
  --epochs 1 \
  --batch-size 2 \
  --crop-size 64 \
  --base-channels 8 \
  --max-images 8
```

The training directory contains the resolved configuration, JSONL loss history, and checkpoints.

## Deep-model inference

```bash
defusion-fuse \
  --image-a path/to/source_a.png \
  --image-b path/to/source_b.png \
  --method cud \
  --checkpoint runs/cud/final.pt \
  --output runs/inference/fused.png
```

Randomly initialized deep-model output is intentionally blocked because it is not a valid result.

## Batch evaluation

Place matched filenames in two directories and run:

```bash
python scripts/evaluate_pairs.py \
  --source-a data/evaluation/source_a \
  --source-b data/evaluation/source_b \
  --output-dir runs/evaluation
```

The script saves every fused image, per-pair CSV metrics, and mean metrics by method. The current
script evaluates classical baselines; evaluate the trained CUD checkpoint using the same image
pairs and extend the table only after the checkpoint and environment are fixed.

## Verification

```bash
ruff check .
pytest --cov=defusion_mtech --cov-report=term-missing
```

## Repository structure

```text
app.py                         Streamlit interface
configs/train_cud.json         Paper-aligned default training configuration
docs/                          Method, evaluation, and submission guidance
scripts/                       Demo-pair generation and batch evaluation
src/defusion_mtech/            Reusable fusion package
tests/                         Unit and model smoke tests
REFERENCES.bib                 Primary references for the report
```

## Project documentation

- [Methodology](docs/methodology.md)
- [Evaluation protocol](docs/evaluation_protocol.md)
- [Final-submission corrections](docs/report_corrections.md)

## Attribution

The CUD concept and DeFusion framework were introduced by Pengwei Liang, Junjun Jiang, Xianming
Liu, and Jiayi Ma at ECCV 2022. This repository was independently written for educational use and
does not copy the official implementation. Cite the original paper using `REFERENCES.bib`.

## License

The project code is released under the MIT License. Dataset and model-weight licenses remain the
responsibility of their respective owners.
