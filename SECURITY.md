# Security Policy

## Supported version

Security fixes are applied to the latest revision on the `main` branch.

## Reporting a vulnerability

Do not disclose a suspected vulnerability in a public issue. Contact the repository owner through
their GitHub profile and include the affected component, reproduction steps, potential impact, and
any suggested mitigation. Please do not include private datasets, credentials, or sensitive image
content in a report.

## Model and data safety

Only load checkpoints from trusted sources. This project uses PyTorch's restricted
`weights_only=True` loading mode, but checkpoint provenance and integrity should still be verified.
Keep datasets, checkpoints, environment files, and experiment outputs outside version control.
