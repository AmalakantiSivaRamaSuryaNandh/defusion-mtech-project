# Contributing

Thank you for helping improve this educational image-fusion project.

## Development setup

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

## Before opening a pull request

Run the complete local validation suite:

```bash
ruff check .
ruff format --check .
pytest --cov=defusion_mtech --cov-report=term-missing
```

Keep each pull request focused. Explain the motivation, list the validation performed, and disclose
any effect on model architecture, training data, preprocessing, metrics, or reported results.

## Research integrity

- Do not commit datasets or model weights unless their licenses permit redistribution.
- Do not add performance claims without the dataset, pair count, protocol, hardware, checkpoint,
  commit, per-pair measurements, mean, and standard deviation.
- Distinguish this independent implementation from the official DeFusion implementation.
- Preserve third-party citations and licenses.
