# ───────────────────────────────────────────────────────────────────
# 1️⃣ Build OpenWhisk Proxy
# ───────────────────────────────────────────────────────────────────
FROM golang:1.23-alpine AS builder_source

ARG GO_PROXY_GITHUB_USER=apache
ARG GO_PROXY_GITHUB_BRANCH=master

# Install dependencies and build OpenWhisk proxy
RUN apk add --no-cache git && \
    git clone --branch ${GO_PROXY_GITHUB_BRANCH} \
    https://github.com/${GO_PROXY_GITHUB_USER}/openwhisk-runtime-go /src && \
    cd /src && \
    env GO111MODULE=on CGO_ENABLED=0 go build main/proxy.go && \
    mv proxy /bin/proxy

# ───────────────────────────────────────────────────────────────────
# 2️⃣ Base CUDA Image for PyTorch + OpenWhisk
# ───────────────────────────────────────────────────────────────────
FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04 AS base

# Install Python & Pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-setuptools && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/local/bin/python && \
    python3 -m pip install --no-cache-dir --upgrade pip

# Install PyTorch (No torchvision)
RUN python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu117

# ───────────────────────────────────────────────────────────────────
# 3️⃣ Copy OpenWhisk Proxy from Builder
# ───────────────────────────────────────────────────────────────────
COPY --from=builder_source /bin/proxy /bin/proxy

# ───────────────────────────────────────────────────────────────────
# 4️⃣ Install OpenWhisk Dependencies
# ───────────────────────────────────────────────────────────────────
COPY requirements_common.txt requirements_common.txt
COPY requirements.txt requirements.txt

# Install Python dependencies for OpenWhisk
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# ───────────────────────────────────────────────────────────────────
# 5️⃣ Set Up OpenWhisk Runtime (Keep /action Directory)
# ───────────────────────────────────────────────────────────────────
# Create and set `/action` as the working directory
RUN mkdir -p /action
WORKDIR /action

# Copy OpenWhisk scripts
COPY bin/compile /bin/compile
COPY lib/launcher.py /lib/launcher.py

# Set OpenWhisk environment variables
ENV OW_LOG_INIT_ERROR=1 \
    OW_WAIT_FOR_ACK=1 \
    OW_EXECUTION_ENV=openwhisk/action-python-cuda \
    OW_COMPILER=/bin/compile

# ───────────────────────────────────────────────────────────────────
# 6️⃣ Final Test Command: Run nvidia-smi & Check PyTorch GPU
# ───────────────────────────────────────────────────────────────────
CMD bash -c "nvidia-smi && python3 -c 'import torch; print(torch.cuda.is_available())'"

# Set OpenWhisk Proxy as Entrypoint
ENTRYPOINT ["/bin/proxy"]
