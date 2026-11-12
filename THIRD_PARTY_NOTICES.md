# Attribution and licenses

FastSET code, author-generated numerical measurements, propagation coefficient
arrays, and model weights in this release are distributed under the repository's
MIT license. The original FastSET/SwiftSET implementation files are identified by
hash in `artifacts/source_provenance.json`.

TensorFlow, NVIDIA Sionna, NumPy, Matplotlib, and their dependencies are installed
from their distributions and retain their own licenses and notices. The receiver
uses Sionna's OFDM, channel, mapping, and 5G LDPC components. RT propagation was
generated using NVIDIA Sionna's ray tracer. Scene meshes and third-party scene
textures are not redistributed. The included RT arrays are the authors' simulated
channel responses used by the experiments; geometry files and images from other
projects are not required to run the released frozen-channel experiments.

The paper's city illustrations are regenerated from the authors' schematic
plotting coordinates. They are not a redistribution of city scene meshes.
