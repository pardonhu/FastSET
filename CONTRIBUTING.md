# Contributing

Please include the release tag, experiment ID, configuration, dependency versions,
GPU model, command, and relevant error text in a reproducibility report. Remove
private paths, access tokens, and unrelated data before posting logs.

For code changes, run `python -m pytest`, `python scripts/audit_release.py`, and
`python -m black --check src scripts tests`. Changes to a scientific protocol
should introduce an explicit configuration and explain the comparison affected.
Archived measurements are immutable; new measurements belong to a new snapshot.
GPU acceptance and CPU tests are reported separately.
