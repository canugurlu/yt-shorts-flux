# Runpod Serverless Dockerfile for FLUX.1-dev
# YouTube Shorts T2I Generator

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST="8.0+PTX;8.6;8.9;9.0" \
    FORCE_CUDA="1" \
    CUDA_VISIBLE_DEVICES=0

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace

# Upgrade pip and install build dependencies
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
RUN pip3 install \
    diffusers==0.30.0 \
    transformers==4.46.0 \
    accelerate==1.0.1 \
    sentencepiece==0.2.0 \
    protobuf==5.28.0 \
    cbor2==1.9.0 \
    huggingface-hub==0.26.2 \
    safetensors==0.4.5 \
    pillow==10.4.0 \
    numpy==1.26.4 \
    compel==2.0.2 \
    t5==0.2.1

# Install Runpod serverless SDK
RUN pip3 install runpod

# Copy handler
COPY handler.py /workspace/handler.py

# Download model weights at build time (optional - can also load at runtime)
# This increases image size but reduces cold start time
# Uncomment if you want to bundle the model in the image
# RUN python3 -c "from diffusers import FluxPipeline; FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype=torch.bfloat16, use_safetensors=True)"

# Set the handler as the entrypoint
ENV HANDLER="handler.py"
ENV RUNPOD_SERVERLESS_TIMEOUT=120

# Runpod serverless will start the handler automatically
CMD ["python3", "-u", "handler.py"]
