# Use runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 as base image
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Configure image maintainer
LABEL maintainer="Nicklas373 <herlambangdicky5@gmail.com>"
LABEL version="1.1.6-PROD"
LABEL description="Docker container for Runpod, used for LLM Quantization with LLM Compressor (AWQ)"

# Configure environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0;12.0"
ENV UV_PYTHON_PREFERENCE=only-system
ENV UV_HTTP_TIMEOUT=1800

# VS Code Server
ENV PASSWORD=""
ENV CODE_SERVER_ARGS="--bind-addr 0.0.0.0:8080 --auth none"

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    git-lfs \
    wget \
    ca-certificates \
    dumb-init \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install code-server
RUN curl -fsSL https://code-server.dev/install.sh | sh

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

# Copy quantization scripts into the container
COPY quantize.py /workspace/
COPY upload.py /workspace/

# Expose VS Code port
EXPOSE 8080

# Set entrypoint and default command
ENTRYPOINT ["dumb-init", "--"]
CMD ["bash", "-c", "code-server $CODE_SERVER_ARGS /workspace & python3 quantize.py --help && sleep infinity"]