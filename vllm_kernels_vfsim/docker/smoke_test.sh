#!/usr/bin/env bash
# One tiny ASC and PTO case under msprof op simulator. No real NPU required.
set -euo pipefail

CANN_SET_ENV="$(ls -d /usr/local/Ascend/cann-*/set_env.sh 2>/dev/null | sort | tail -1 || true)"
if [[ -n "${CANN_SET_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${CANN_SET_ENV}"
fi
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib:${LD_LIBRARY_PATH:-}"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "${ROOT}/cases/kv_index_cat/run.py" ]]; then
  # Installed copy lives next to the image WORKDIR checkout.
  ROOT="/workspace/vllm_kernels_vfsim"
fi
cd "${ROOT}"
python cases/kv_index_cat/run.py asc
python cases/kv_index_cat/run.py pto
echo "smoke ok"
