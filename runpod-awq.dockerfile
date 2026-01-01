# Use runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 as base image
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Configure image maintainer
LABEL maintainer="Nicklas373 <herlambangdicky5@gmail.com>"
LABEL version="1.1.4-PROD"
LABEL description="Docker container for Runpod, used for LLM Quantization with LLM Compressor (AWQ)"

# Configure environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0;12.0"
ENV UV_PYTHON_PREFERENCE=only-system
ENV UV_HTTP_TIMEOUT=1800

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    git-lfs \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install uv for python virtual environments
RUN curl -LsSf https://astral.sh/uv/install.sh | bash
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# Configure workspace
WORKDIR /workspace

# Create virtual environment
RUN uv venv /workspace/.venv
ENV PATH="/workspace/.venv/bin:${PATH}"

# Copy requirements file into the container
COPY requirements.txt /workspace/requirements.txt

# Install other required Python packages
RUN uv pip install --no-cache-dir -r /workspace/requirements.txt

# Install VLLM for evaluation benchmark
RUN uv pip install -U vllm --torch-backend=auto --extra-index-url https://download.pytorch.org/whl/cu128

# Copy quantization scripts into the container
COPY quantize.py /workspace/
COPY upload.py /workspace/

# Set entrypoint and default command
ENTRYPOINT ["bash", "-c", "set -e; ls; python3 quantize.py --help; sleep infinity"]