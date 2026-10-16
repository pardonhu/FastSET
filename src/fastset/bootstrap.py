import os
from pathlib import Path
from .common import save


def initialize_gpu(seed, out):
    os.environ["MPLCONFIGDIR"] = str(out / "mpl_cache")
    from fastset.runtime import import_tensorflow_for_bootstrap, seed_and_enable_determinism

    tf, requested_determinism = import_tensorflow_for_bootstrap()
    devices = tf.config.list_physical_devices("GPU")
    env = dict(
        tensorflow=tf.__version__,
        physical_gpus=[str(d) for d in devices],
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        requested_tf_deterministic_ops=requested_determinism,
    )
    save(out / "environment.json", env)
    if not devices:
        raise RuntimeError("GPU unavailable; neural CPU fallback is forbidden")
    tf.config.experimental.set_memory_growth(devices[0], True)
    with tf.device("/GPU:0"):
        probe = tf.matmul(tf.ones([2, 2]), tf.ones([2, 2]))
        assert "GPU" in probe.device and float(tf.reduce_sum(probe)) == 8
    import sionna

    env.update(
        sionna=sionna.__version__,
        probe_device=probe.device,
        cuda_libraries=sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "libcuda" in line
            }
        ),
    )
    with tf.device("/GPU:0"):
        probe = tf.matmul(tf.ones([2, 2]), tf.ones([2, 2]))
        assert "GPU" in probe.device and float(tf.reduce_sum(probe)) == 8
    env["post_sionna_device"] = probe.device
    save(out / "environment.json", env)
    env["rng_bootstrap"] = seed_and_enable_determinism(tf, sionna, seed, requested_determinism)
    save(out / "environment.json", env)
    return tf, sionna
