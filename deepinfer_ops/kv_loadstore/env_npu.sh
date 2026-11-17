#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /home/jzhuang/miniconda/etc/profile.d/conda.sh
conda activate cann91_dev
set +u
source /usr/local/Ascend/canns/9.2.0/cann-9.2.0/set_env.sh
set -u
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-${ROOT}/.tilelang_cache_kv}"
export CCACHE_DIR="${CCACHE_DIR:-/home/jzhuang/work_dir/.ccache_user}"
export PTO_FLAGS="${PTO_FLAGS:---pto-backend=vpto --pto-level=level3 --vmi-layout-max-frontier-entries=65536 --vmi-layout-max-transitions=262144 --enable-vecscope-mem-bar=false}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
exec "$@"
