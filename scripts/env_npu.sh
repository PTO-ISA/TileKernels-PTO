#!/usr/bin/env bash
# Activate cann91_dev + CANN 9.2.0 and run a command with this worktree on PYTHONPATH.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source /home/jzhuang/miniconda/etc/profile.d/conda.sh
conda activate cann91_dev
# shellcheck disable=SC1091
source /usr/local/Ascend/canns/9.2.0/cann-9.2.0/set_env.sh
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-/home/jzhuang/work_dir/.tilelang_cache}"
export CCACHE_DIR="${CCACHE_DIR:-/home/jzhuang/work_dir/.ccache}"
# level3 matches the TileKernels-vmi on-device runs. The default VMI layout
# frontier (288) cannot schedule the norm VF; raise it so the ASC schedule
# still lowers. Kernels that fit in 288 stop at the same layout.
# --enable-vecscope-mem-bar=false stops ptoas 0.72 from inserting a
# VST_VST barrier after every versioned-buffer store; the kernels carry no
# real UB hazard that needs it, and the barrier cost rows 0.3-0.7x of ASC.
export PTO_FLAGS="${PTO_FLAGS:---pto-backend=vpto --pto-level=level3 --vmi-layout-max-frontier-entries=65536 --vmi-layout-max-transitions=262144 --enable-vecscope-mem-bar=false}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
exec "$@"
