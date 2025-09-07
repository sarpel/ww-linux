# Dockerfile for Wake Word Generator - Linux Edition
# Design v4: Added symbolic link to runtime stage.

# ==============================================================================
# Builder Stage
# ==============================================================================
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.10 and other build-time dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
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
RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Create a virtual environment with Python 3.10.
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .

RUN pip install -r requirements.txt

# ==============================================================================
# Runtime Stage
# ==============================================================================
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Update apt sources to jammy
RUN sed -i 's/focal/jammy/g' /etc/apt/sources.list && \
# Install Python 3.10 and other build-time dependencies.
    apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
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

WORKDIR /app

# Create a virtual environment with Python 3.10.
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .

RUN pip install -r requirements.txt

# ==============================================================================
# Runtime Stage
# ==============================================================================
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app/src

# Update apt sources to jammy
RUN sed -i 's/focal/jammy/g' /etc/apt/sources.list && \
# Install Python 3.10 and other runtime dependencies.
    apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    libasound2 \
    libsndfile1 \
    libfftw3-3 \
    libatlas3-base \
    ffmpeg \
    sox \
    curl \
    && rm -rf /var/lib/apt/lists/*




ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app/src

# Install Python 3.10 and other runtime dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    libasound2 \
    libsndfile1 \
    libfftw3-3 \
    libatlas3-base \
    ffmpeg \
    sox \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a symbolic link for python to ensure consistency for the entrypoint.
RUN ln -s /usr/bin/python3.10 /usr/bin/python


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
