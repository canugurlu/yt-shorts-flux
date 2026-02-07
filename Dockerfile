# Runpod Serverless Dockerfile for FLUX.1-dev
# YouTube Shorts T2I Generator

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Hugging Face token will be passed as build arg
ARG HF_TOKEN

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_VISIBLE_DEVICES=0 \
    HF_HOME=/workspace/models \
    PYTHONWARNINGS="ignore::FutureWarning"

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create working directory with model cache
WORKDIR /workspace

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch with CUDA 12.1 support (smaller package)
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies (without strict versions for compatibility)
RUN pip3 install \
    diffusers \
    transformers \
    accelerate \
    sentencepiece \
    huggingface-hub \
    safetensors \
    pillow \
    numpy \
    compel \
    protobuf \
    cbor2 \
    bitsandbytes

# Login to Hugging Face (required for gated models)
RUN huggingface-cli login --token ${HF_TOKEN}

# Install Runpod serverless SDK
RUN pip3 install runpod

# Copy handler
COPY handler.py /workspace/handler.py

# Set the handler as the entrypoint
ENV HANDLER="handler.py"

# Runpod serverless will start the handler automatically
CMD ["python3", "-u", "handler.py"]
