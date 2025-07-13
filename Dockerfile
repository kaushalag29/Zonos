# Base image with Anaconda
FROM continuumio/miniconda3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DOCKER=true

# Set working directory
WORKDIR /app

# Install system dependencies
# espeak-ng is required for phonemization
# git is needed for some pip installs from git repos
# ffmpeg is a common dependency for audio processing
# rustc and cargo are needed for building sudachipy
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    espeak-ng \
    curl \
    build-essential \
    && apt-get clean

# Install Rust (required for sudachipy compilation)
RUN curl --proto '=https' --tlsv1.3 https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Copy the application files into the container
COPY . .

# Create a conda environment for the application
# Zonos requires Python >= 3.10, using 3.11 for consistency
RUN conda create -n zonos python=3.11 -y

# Activate the conda environment for subsequent commands
SHELL ["conda", "run", "-n", "zonos", "/bin/bash", "-c"]

# Install uv for faster package management
RUN pip install uv

# Upgrade pip to get latest wheel support
RUN pip install --upgrade pip

# Install the project and its dependencies from pyproject.toml
# Install only base dependencies (compile extras require CUDA compilation)
RUN uv pip install --system -e .

# Pre-download the Zonos model during build time
# This ensures the model is cached in the Docker image and doesn't need to be downloaded at runtime
RUN python -c "from zonos.model import Zonos; Zonos.from_pretrained('Zyphra/Zonos-v0.1-transformer', device='cpu')"

# Expose the port the server will run on (port 8011 as specified in server.py)
EXPOSE 8011

# Command to run the application server with conda environment activated
CMD ["conda", "run", "-n", "zonos", "python", "server.py"]
