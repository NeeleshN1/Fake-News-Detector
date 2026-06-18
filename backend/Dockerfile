FROM ubuntu:22.04

# Disable interactive prompts (fixes tzdata hang)
ENV DEBIAN_FRONTEND=noninteractive

# Build + runtime deps (ONNX Runtime is fetched as a prebuilt release below)
RUN apt-get update && apt-get install -y \
    g++ \
    cmake \
    python3 \
    python3-pip \
    curl \
    ca-certificates \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# ONNX Runtime (prebuilt, CPU)
# -------------------------
ARG ORT_VERSION=1.20.1
RUN curl -fsSL -o /tmp/ort.tgz \
      https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm /tmp/ort.tgz
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib

WORKDIR /app
COPY . .

# Build C++ backend against ONNX Runtime
RUN cmake -S . -B build -DORT_ROOT=/opt/onnxruntime \
    && cmake --build build --config Release \
    && find build -type f -executable -name "fake_news" -exec cp {} ./fake_news \;

# Python API deps
RUN pip3 install --no-cache-dir fastapi uvicorn

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
