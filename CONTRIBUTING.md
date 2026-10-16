# Contributing

Install the CPU test dependencies and run `python -m pytest` and
`python scripts/audit_release.py`. Include the command, config, manifest hash,
package versions and a small log excerpt in a bug report. Never attach passwords,
SSH keys, full private paths, or unpublished datasets.

Changes to channel parameters, labels, optimizer settings, batch order, stopping
criteria or timing scope require an explicit protocol change, separate result
directories and updated documentation. Do not replace historical measured CSVs
with regenerated values without recording their provenance.

Numerical model changes need a GPU comparison of initial weight shapes/order,
weight load/readback, forward output, loss/update, decoded bits and integer error
counts. Wall-time claims require the same device and declared measurement scope.
CPU plotting/tests do not establish GPU numerical equivalence.

Report failed and non-converged configurations. Do not tune thresholds or choose
checkpoints after inspecting the held-out evaluation set.

