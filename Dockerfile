FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch==2.3.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir \
    transformers \
    accelerate \
    safetensors \
    httpx \
    orjson \
    pyyaml \
    numpy \
    pyarrow \
    "sglang[all]>=0.5.6"

WORKDIR /app
COPY run_nla.py test.txt ./
