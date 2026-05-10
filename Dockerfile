###############################################################################
# PY-IDR container image — JAX/CUDA runtime for the simulation sweep + real-data
# fits on the user's a10-dev host (4× A10 GPUs) and on GitHub's GPU runners.
#
# Build: docker build -t pyidr:dev .
# Run:   bash scripts/run_on_a10dev.sh doctor   (see plans/14_docker_a10dev.md)
###############################################################################

# ---- builder stage ---------------------------------------------------------------
ARG CUDA_BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_BASE} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        build-essential \
        python3.11 \
        python3.11-venv \
        python3-pip \
 && rm -rf /var/lib/apt/lists/*

# uv for fast wheel resolution.
RUN pip install --no-cache-dir uv

WORKDIR /workspace
COPY pyproject.toml /workspace/pyproject.toml
COPY README.md /workspace/README.md
COPY src/py_idr/__init__.py /workspace/src/py_idr/__init__.py
COPY src/py_idr/_version.py /workspace/src/py_idr/_version.py

# Install the package + the GPU/data/viz/dev extras into a dedicated venv.
# We install jax[cuda12] explicitly via the JAX wheel index to pin the CUDA build.
RUN uv venv --python 3.11 /opt/venv \
 && /opt/venv/bin/python -m pip install --upgrade pip wheel \
 && /opt/venv/bin/python -m pip install --no-cache-dir -e ".[dev,data,viz]" \
 && /opt/venv/bin/python -m pip install --no-cache-dir \
        --upgrade "jax[cuda12]" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Copy the rest of the source after deps are installed so source-only changes
# don't bust the dependency layer.
COPY . /workspace

# ---- runtime stage ---------------------------------------------------------------
FROM ${CUDA_BASE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:${PATH} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Mandatory environment variables for multi-GPU JAX.
# - PREALLOCATE=false stops one process from grabbing all 24 GB on import.
# - X64=1 matches our test/conftest defaults.
ENV JAX_PLATFORMS=cuda \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_ENABLE_X64=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3.11 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /workspace /workspace

ENTRYPOINT ["python", "-m", "py_idr.cli"]
CMD ["doctor"]
