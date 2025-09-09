# this docker image is built based on Dockerfile_base
FROM us-central1-docker.pkg.dev/gpu-launchpad-playground/tiangel-customer-workshop/llama-base:v0.1

# Installation arguments
ARG PIP_INDEX=https://pypi.org/simple
ARG EXTRAS=metrics
ARG INSTALL_FLASHATTN=false
ARG HTTP_PROXY=""

# Define environments
ENV MAX_JOBS=16
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""
ENV PIP_ROOT_USER_ACTION=ignore

RUN pip install --upgrade torch torchvision torchaudio flashinfer-python

WORKDIR /

COPY trainer /trainer

COPY LLaMA-Factory /LLaMA-Factory

COPY requirements.txt /requirements.txt

# Run all commands in a single layer
RUN pip install -r requirements.txt && \
    pip install -e "/LLaMA-Factory[torch,metrics,deepspeed]" --no-build-isolation

# Rebuild flash attention
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
    fi

# Default entrypoint, could be replaced in custom-training.py
ENTRYPOINT ["python", "-m", "trainer.run_train_v2.py","/gcs/salander-us-central1/llama-factory/config/llama3_lora_sft-Copy1.yaml"]
