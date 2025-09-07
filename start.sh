#!/bin/bash
# Wake Word Generator - Linux Ubuntu Edition
# Quick start script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_error "Virtual environment not found!"
    print_status "Please run './install.sh' first to set up the environment."
    exit 1
fi

# Check for GPU
print_status "Checking system status..."
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader,nounits | head -1
else
    print_warning "No NVIDIA GPU detected. Training will not be available."
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Check Python packages
print_status "Checking Python packages..."
python -c "
import torch
import tensorflow as tf
print('🔥 PyTorch version:', torch.__version__)
print('🔥 PyTorch CUDA available:', torch.cuda.is_available())
print('🔥 TensorFlow version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('🔥 TensorFlow GPUs:', len(gpus))
"

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

print_status "Environment variables set for optimal performance"

# Create data directories if they don't exist
mkdir -p data/{training,models,generate,augment}

# Check port availability
PORT=${1:-7860}
if lsof -i:$PORT &> /dev/null; then
    print_warning "Port $PORT is already in use. The application will find an available port automatically."
fi

print_success "Starting Wake Word Generator - Linux Ubuntu Edition..."
print_status "The web interface will be available at http://localhost:$PORT"
print_status "Press Ctrl+C to stop the application"

# Start the application
python app.py

print_status "Application stopped."