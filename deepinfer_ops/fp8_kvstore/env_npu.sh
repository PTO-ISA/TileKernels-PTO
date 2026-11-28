#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${CONDA_SH:-}" ]]; then
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
fi
CONDA_ENV="${CONDA_ENV:-cann91_dev}"
CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/canns/9.2.0/cann-9.2.0/set_env.sh}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set +u
source "${CANN_SET_ENV}"
set -u
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-${ROOT}/.tilelang_cache_fp8}"
export CCACHE_DIR="${CCACHE_DIR:-${ROOT}/.ccache_fp8}"
export PTO_FLAGS="${PTO_FLAGS:---pto-backend=vpto --pto-level=level3 --vmi-layout-max-frontier-entries=65536 --vmi-layout-max-transitions=262144 --enable-vecscope-mem-bar=false}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
exec "$@"
