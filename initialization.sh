#!/bin/bash
# Prepares the lab. The Makefile runs it before every build, so you do not need to run it by hand.
# It only acts when something is missing, so running it again is harmless.

(
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
LAB_DIR=$(pwd)

# Download PULP-TrainLib
if [ ! -d pulp-trainlib/lib ]; then
    git submodule update --init --recursive
fi

# Fix PULP-TrainLib for the RISC-V toolchain of the devcontainer
if git -C pulp-trainlib apply --check "$LAB_DIR/patch/pulp-trainlib.patch" 2>/dev/null; then
    echo "Patching PULP-TrainLib"
    git -C pulp-trainlib apply "$LAB_DIR/patch/pulp-trainlib.patch"
fi

# Lab machines (virtual machine) only: enlarge the L2 memory of the simulated PULP chip.
# The devcontainer simulator already has enough L2 memory.
VM_PULP_JSON=/workspaces/APAI-Docker/pulp-sdk/tools/gap-configs/configs/chips/pulp/pulp.json
if [ -f "$VM_PULP_JSON" ] && ! cmp -s patch/pulp.json "$VM_PULP_JSON"; then
    echo "Patching the PULP SDK memory configuration"
    cp patch/pulp.json "$VM_PULP_JSON"
fi
)
