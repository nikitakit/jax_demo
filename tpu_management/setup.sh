#!/bin/bash
# Setup script for TPU VMs
# "tpu.sh provision" and "pod.sh provision" will run this file on the TPU VMs

# Set up PATH
export PATH="$HOME/.local/bin:$PATH"

# Set up SSH across TPU vms
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

# Install key dependencies
pushd jax_demo && git checkout pushbranch && git checkout -b main && popd
pip3 install --user --upgrade -r jax_demo/requirements_tpu.txt

# Install Flax
git clone https://github.com/google/flax.git
pip install --user -e flax

# JAX pod setup helper (only run in a multi-host TPU setup)
# Running the alias "jax_pod_setup" in all worker VMs will cause subsequent
# JAX-based programs to use all workers for multi-host training.
if [ -f ./jax_pod_setup.sh ]; then
    chmod +x ./jax_pod_setup.sh
    printf 'alias jax_pod_setup="eval \\\"$(~/jax_pod_setup.sh)\\\""' >> ~/.bash_aliases
fi

