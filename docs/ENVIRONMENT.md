# Environment and installation

## Tested local reference

- Python 3.10
- TensorFlow 2.15.1 / Sionna 0.19.2 / NumPy 1.26.4
- Linux/WSL, NVIDIA RTX 3090 Ti

GPU execution has two levels of status: the original campaign used this
environment; the reorganized command wrappers still need their own numerical
acceptance. CPU tests and wheel installation are checked separately.

The CPU extra installs plotting/test dependencies only. The `gpu` extra adds
the pinned TensorFlow/Sionna stack, including user-environment CUDA libraries.
Do not upgrade to Sionna 1.x or TensorFlow 2.16+ without a new port and numerical
comparison. NumPy 2.x is not a supported replacement for the legacy array
serialization path; the package pins NumPy 1.26.4.

Use a new conda/venv environment. Installation does not modify the host driver.
If TensorFlow cannot create a real GPU device, the neural runner fails instead
of silently reporting CPU results as GPU measurements. A newer system driver
may support the user-space CUDA stack; a driver downgrade is not part of this
artifact's installation instructions.

Remote historical runs also used a TensorFlow 2.14 environment for driver
compatibility. Their metadata and original results should remain distinguished
from fresh 2.15.1 runs. A seed does not guarantee bit-identical GPU arithmetic
across library versions or devices.

## Deterministic startup

Sionna 0.19.2 initializes its TensorFlow generator lazily. The startup helper
creates and seeds that generator on CPU through the public API **before**
enabling deterministic ops, then enables determinism for model computation.
The environment's determinism request is deferred only before TensorFlow import;
it is not globally disabled to bypass an error. Importing TensorFlow before
the helper is an explicit error.

## Packaging conventions

The project uses a `src/` layout, `pyproject.toml`, optional GPU dependencies,
and a console entry point. These follow the
[Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/).
The repository archive contains configs, data cards and measured artifacts;
the Python wheel contains the executable package. Use the repository archive
for paper reproduction, not the wheel alone.

