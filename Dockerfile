FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

RUN python -m pip install numpy>=1.26.4 numba>=0.60.0 transformers>=4.44.2 \
    bitsandbytes accelerate sentencepiece protobuf einops

# ビルドできなくてワロタ決め打ち
RUN python -m pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post7/causal_conv1d-1.5.0.post7+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN python -m pip install https://github.com/state-spaces/mamba/releases/download/v1.2.0.post1/mamba_ssm-1.2.0.post1+cu121torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

COPY . .

CMD ["/bin/bash"]