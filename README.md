# Presentation on JAX

## Notes for TPU

Scripts in `tpu_management/` are documented at https://github.com/nikitakit/sabertooth#automatic-installation-in-tpu-vms

TPU quickstart: https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm
TPU console: https://console.cloud.google.com/compute/tpus

## Notes for CPU

```
conda create -n jax_demo python=3.8.5
conda activate jax_demo
pip install torch==1.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install ipython ipykernel ipywidgets matplotlib
pip install jax jaxlib
```

To emulate multi-device training with 8 CPU "devices", run the following before importing jax.
```
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
```

## Notes for GPU

See https://github.com/google/jax#installation and the CPU commands above.

## More links

- https://github.com/google/jax
- https://github.com/google/flax
- https://github.com/deepmind/optax
