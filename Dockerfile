# Dockerfile for Wake Word Generator - Linux Edition
# Design v4: Added symbolic link to runtime stage.

# ==============================================================================
# Builder Stage
# ==============================================================================
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.9 and other build-time dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3-pip \
    build-essential \
    cmake \
    pkg-config \
    libasound2-dev \
    portaudio19-dev \
    libsndfile1-dev \
    libfftw3-dev \
    libatlas-base-dev \
    gfortran \
    git \
    curl \
    wget \
    unzip \
    ffmpeg \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Create a symbolic link for python to ensure consistency.
RUN ln -s /usr/bin/python3.9 /usr/bin/python

WORKDIR /app

# Create a virtual environment with Python 3.9.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .

RUN pip install -r requirements.txt

# ==============================================================================
# Runtime Stage
# ==============================================================================
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app/src

# Install Python 3.9 and other runtime dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    libasound2 \
    libsndfile1 \
    libfftw3-3 \
    libatlas3-base \
    ffmpeg \
    sox \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a symbolic link for python to ensure consistency for the entrypoint.
RUN ln -s /usr/bin/python3.9 /usr/bin/python

RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin -c "Application User" appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY . .

RUN mkdir -p data/training data/models data/generate data/augment

RUN chown -R appuser:appgroup /app

USER appuser

LABEL maintainer="Your Name <you@example.com>"
LABEL description="Wake Word Generator - Linux Edition. For creating and training custom wake words."

EXPOSE 7860

CMD ["python", "app.py"]
