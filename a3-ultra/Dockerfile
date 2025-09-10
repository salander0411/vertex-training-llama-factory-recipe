ARG PYTORCH_IMAGE
# Stage 1: Get the NCCL plugin
FROM us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib:v1.0.6 AS nccl_plugin

# Stage 2: Main build
# Using a specific PyTorch version as the base
FROM nvcr.io/nvidia/pytorch:25.03-py3

# --- ARG and ENV Declarations ---
# All arguments and environment variables are grouped at the top for clarity.
ARG LLAMA_FACTORY_VERSION="main"
ARG APT_SOURCE=https://mirrors.tuna.tsinghua.edu.cn/ubuntu/
# Add a specific ARG for the mirror
ARG PIP_MIRROR_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
ARG PIP_INDEX=https://pypi.org/simple
ARG EXTRAS=metrics,deepspeed,liger-kernel
ARG INSTALL_FLASHATTN=false
ARG HTTP_PROXY=""

ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=16
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV PIP_ROOT_USER_ACTION=ignore
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV http_proxy="${HTTP_PROXY}"
ENV https_proxy="${HTTP_PROXY}"

# Set the working directory
WORKDIR /app

# --- System Dependencies ---
# Step 3: Set APT source
RUN echo "Setting APT source to ${APT_SOURCE}" && \
    cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
    { \
      echo "deb ${APT_SOURCE} jammy main restricted universe multiverse"; \
      echo "deb ${APT_SOURCE} jammy-updates main restricted universe multiverse"; \
      echo "deb ${APT_SOURCE} jammy-backports main restricted universe multiverse"; \
      echo "deb ${APT_SOURCE} jammy-security main restricted universe multiverse"; \
    } > /etc/apt/sources.list

# Step 4: Install system utilities and build tools
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      # System utilities
      git wget vim \
      # Build tools
      gcc g++ \
      # Google Cloud SDK dependencies
      ca-certificates curl gnupg cmake dnsutils pciutils rsync && \
    # Clean up APT cache in the same layer to reduce image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Step 5: Install Google Cloud SDK
RUN --mount=type=cache,target=/var/cache/apt \
    echo "deb https://packages.cloud.google.com/apt gcsfuse-buster main" | tee /etc/apt/sources.list.d/gcsfuse.list && \
    echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl -sS https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y google-cloud-cli && \
    # Clean up APT cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Step 6: Install pre-built wheels using a mirror for speed
RUN pip config set global.index-url "${PIP_MIRROR_INDEX}" && \
    pip install --no-cache-dir --upgrade pip packaging wheel setuptools && \
    # Install flashinfer from pre-built wheel
    wget -nv https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl && \
    pip install --no-cache-dir flashinfer_python-0.2.5+cu128torch2.7-cp38-abi3-linux_x86_64.whl && \
    rm flashinfer_python-0.2.5+cu128torch2.7-cp38-abi3-linux_x86_64.whl && \
    # Reset pip config to avoid using the mirror for subsequent steps
    pip config unset global.index-url

# --- Application Setup ---
# Step 7: Clone the repository
RUN git clone https://github.com/hiyouga/LLaMA-Factory.git /app/LLaMA-Factory && \
    cd /app/LLaMA-Factory && \
    git checkout ${LLAMA_FACTORY_VERSION}

# Step 8: Install application dependencies
RUN cd /app/LLaMA-Factory && \
    unset PIP_CONSTRAINT && \
    pip install "dill==0.3.8" && \
    pip config set global.index-url "${PIP_INDEX}" && \
    # Install application requirements
    # remove version restriction on datasets package to resolve conflict
    # sed -i 's/^datasets.*/datasets/' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e ".[torch,${EXTRAS}]" --no-build-isolation && \
    pip install retrying absl-py

# Step 9: Rebuild flash attention
RUN pip uninstall -y transformer-engine flash-attn && \
    if [ "$INSTALL_FLASHATTN" == "true" ]; then \
        pip uninstall -y ninja && \
        if [ -n "$HTTP_PROXY" ]; then \
            pip install --proxy=$HTTP_PROXY ninja && \
            pip install --proxy=$HTTP_PROXY --no-cache-dir flash-attn --no-build-isolation; \
        else \
            pip install ninja && \
            pip install --no-cache-dir flash-attn --no-build-isolation; \
        fi; \
    fi && \
    # Reset pip config
    pip config unset global.index-url

# --- Final Steps ---
# Step 10: Copy NCCL plug-in libraries from the first stage
COPY --from=nccl_plugin /usr/local/gib /usr/local/gib
COPY --from=nccl_plugin /scripts /scripts
RUN /scripts/container_entry.sh install --install-nccl

# Install NCCL dependencies
RUN apt upgrade -y --allow-change-held-packages libnccl2=2.25.1-1+cuda12.8 libnccl-dev=2.25.1-1+cuda12.8

# Step 11: Copy torchrun launcher script
COPY trainer/ /app/trainer/