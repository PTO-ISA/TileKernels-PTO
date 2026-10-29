# Self-contained image for the DeepInfer SimdVF ASC->PTO ports.
# Toolchain pin matches the Ascend puzzle container:
#   CANN 9.2.0-beta.2, torch 2.9.0 (CPU wheel), torch-npu 2.9.0.post6,
#   ptoas 0.70, tilelang commit 8e10a859 (pto-dev snapshot).
# Agent CLIs from that image are omitted; they are not needed to reproduce kernels.
#
# Build from this directory's parent (vllm_kernels_vfsim/):
#   docker build -f docker/Dockerfile .

FROM quay.io/ascend/cann:9.2.0-beta.2-950-ubuntu22.04-py3.12

RUN apt-get update && apt-get install -y \
    wget curl git vim gawk ripgrep \
    ninja-build cmake ccache build-essential lld zip binutils patchelf chrpath \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir torch-npu==2.9.0.post6 \
    && pip install --no-cache-dir numpy pyyaml

# LLVM/MLIR Python bindings are not yet compatible with pybind11 3.x
RUN pip install --no-cache-dir pytest 'pybind11==2.12.0'

# Self-contained PTO toolchain from hw-native-sys/PTOAS v0.70 (cp312 matches this image).
RUN set -eux; \
    arch="$(uname -m)"; \
    case "${arch}" in \
      x86_64) \
        PTOAS_ARCH=x86_64; \
        PTOAS_SHA256=42f46a76c13f6830fc65978ca03fba4ac2a194228e22d1c253af813cbe9f331b; \
        ;; \
      aarch64) \
        PTOAS_ARCH=aarch64; \
        PTOAS_SHA256=8607c2da8bf5230ca78b263159a6fd88473d789820532687af42b43a096cdfe3; \
        ;; \
      *) \
        echo "Unsupported architecture for ptoas: ${arch}" >&2; \
        exit 1; \
        ;; \
    esac; \
    PTOAS_FILENAME="ptoas-0.70-cp312-cp312-manylinux_2_34_${PTOAS_ARCH}.whl"; \
    curl --fail --location --retry 3 \
      --output "/tmp/${PTOAS_FILENAME}" \
      "https://github.com/hw-native-sys/PTOAS/releases/download/v0.70/${PTOAS_FILENAME}"; \
    printf '%s  %s\n' "${PTOAS_SHA256}" "/tmp/${PTOAS_FILENAME}" | sha256sum --check -; \
    pip install --no-cache-dir "/tmp/${PTOAS_FILENAME}"; \
    rm -f "/tmp/${PTOAS_FILENAME}"

# Pin a commit on https://github.com/PTO-ISA/tilelang branch pto-dev.
# Pinned at 8e10a859. pto-dev moves; to pick up a newer snapshot, update
# TILELANG_COMMIT by hand. Do not track the branch tip.
ARG TILELANG_REPO=https://github.com/PTO-ISA/tilelang.git
ARG TILELANG_COMMIT=8e10a8592f17913decde3289b887f4bc9e247caf
RUN set -eux; \
    git init /opt/tilelang; \
    git -C /opt/tilelang remote add origin "${TILELANG_REPO}"; \
    git -C /opt/tilelang fetch --depth 1 origin "${TILELANG_COMMIT}"; \
    git -C /opt/tilelang checkout --detach FETCH_HEAD; \
    git -C /opt/tilelang submodule update --init --recursive --depth 1; \
    find /opt/tilelang -name .git -prune -exec rm -rf {} +
ENV USE_CUDA=0 \
    USE_ROCM=0 \
    CMAKE_GENERATOR=Ninja
RUN cd /opt/tilelang \
    && CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" pip install --no-cache-dir -v .

COPY docker/smoke_test.sh /usr/local/bin/vfsim-smoke
RUN chmod +x /usr/local/bin/vfsim-smoke

WORKDIR /workspace
