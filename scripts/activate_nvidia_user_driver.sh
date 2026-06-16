#!/usr/bin/env bash

# Use the user-local NVIDIA driver libraries when the system libraries lag the
# loaded kernel module. nvidia-modprobe is setuid-root on the training host and
# safely creates the UVM device nodes when they are missing after a reboot.
NVIDIA_USER_ROOT="${NVIDIA_USER_ROOT:-/home/hhai/l20-pretrain/.nvidia-user-580.159.04}"
NVIDIA_USER_LIB="$NVIDIA_USER_ROOT/usr/lib/x86_64-linux-gnu"

if [[ ! -r "$NVIDIA_USER_LIB/libcuda.so.1" ]]; then
  echo "Missing user-local NVIDIA driver libraries: $NVIDIA_USER_LIB" >&2
  return 1 2>/dev/null || exit 1
fi

if ! lsmod | grep -q '^nvidia_uvm '; then
  nvidia-modprobe -u -c=0
fi

export PATH="$NVIDIA_USER_ROOT/usr/bin:$PATH"
export LD_LIBRARY_PATH="$NVIDIA_USER_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
