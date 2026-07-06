# Dockerfile for running Galician ASR (wav2vec2 + KenLM)
# Model: proxectonos/Nos_ASR-wav2vec2-large-xlsr-53-gl-with-lm (HuggingFace)

FROM python:3.10-slim-bookworm

ENV VIRTUAL_ENV=/opt/venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential gcc g++ cmake \
        ffmpeg libsndfile1 \
        libboost-system-dev libboost-thread-dev libboost-program-options-dev \
        libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev \
        git curl ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --quiet --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt
RUN pip install --quiet -r /app/requirements.txt \
    && rm -rf /root/.cache/pip

COPY . /app

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models/.hf_cache
