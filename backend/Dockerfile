# ---------- build stage: compile the C++ inference binary ----------
FROM ubuntu:22.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    g++ cmake python3 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime (prebuilt, CPU)
ARG ORT_VERSION=1.20.1
RUN curl -fsSL -o /tmp/ort.tgz \
      https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm /tmp/ort.tgz

WORKDIR /src
COPY . .
RUN cmake -S . -B build -DORT_ROOT=/opt/onnxruntime \
    && cmake --build build --config Release \
    && cp "$(find build -type f -name fake_news -executable | head -1)" /src/fake_news

# ---------- runtime stage: slim image, no compilers ----------
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
# --no-install-recommends avoids pulling gcc/dev (~1GB); libgomp1 is ONNX Runtime's OpenMP dep.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir fastapi uvicorn

WORKDIR /app
# ONNX Runtime shared libs + the compiled binary
COPY --from=build /opt/onnxruntime/lib/ /usr/local/lib/onnxruntime/
COPY --from=build /src/fake_news /app/fake_news
# app code + deployable model artifacts (int8 onnx, vocab, config — per .dockerignore)
COPY api.py /app/api.py
COPY model/ /app/model/

ENV LD_LIBRARY_PATH=/usr/local/lib/onnxruntime
RUN ldconfig /usr/local/lib/onnxruntime

# Cloud Run (and Render) inject $PORT; default to 8080 for local runs.
EXPOSE 8080
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}"]
