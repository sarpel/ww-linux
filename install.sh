#!/bin/bash
# Wake Word Generator - Linux Ubuntu Installation Script
# Tested on Ubuntu 20.04, 22.04

set -e

echo "🚀 Setting up Wake Word Generator for Linux Ubuntu..."

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

# Check if running on Ubuntu
if ! grep -q "Ubuntu\|Debian" /etc/os-release; then
    print_warning "This script is optimized for Ubuntu/Debian. Other distributions may need modifications."
fi

# Update system packages
print_status "Updating system packages..."
sudo apt update

# Install system dependencies
print_status "Installing system dependencies..."
sudo apt install -y \
    python3.10-dev \
    python3.10-pip \
    python3.10-venv \

    build-essential \
    cmake \
    pkg-config \
    libasound2-dev \
    portaudio19-dev \
    libsndfile1-dev \
    libfftw3-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    git \
    curl \
    wget \
    unzip \
    ffmpeg \
    sox

# Check for NVIDIA GPU
print_status "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    
    # Check CUDA installation
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')
        print_success "CUDA $CUDA_VERSION detected"
    else
        print_warning "CUDA not found. Please install CUDA 11.8+ for GPU acceleration"
        print_status "Download from: https://developer.nvidia.com/cuda-downloads"
    fi
else
    print_error "No NVIDIA GPU detected. GPU training will not be available."
    read -p "Continue with CPU-only setup? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
print_status "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3.10 -m venv venv
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install all requirements from the file
print_status "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

# Test GPU availability
print_status "Testing GPU availability..."
python3.10 -c "
import torch
import tensorflow as tf

print('🔥 PyTorch GPU available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('🔥 PyTorch GPU count:', torch.cuda.device_count())
    print('🔥 PyTorch GPU name:', torch.cuda.get_device_name(0))

print('🔥 TensorFlow GPU devices:')
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print('  -', gpu.name)

if torch.cuda.is_available() or len(gpus) > 0:
    print('✅ GPU acceleration ready!')
else:
    print('⚠️  No GPU acceleration available')
"

# Create desktop entry (optional)
read -p "Create desktop shortcut? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cat > ~/.local/share/applications/wakeword-generator.desktop << EOF
[Desktop Entry]
Name=Wake Word Generator
Comment=Generate and train wake word detection models
Exec=$(pwd)/venv/bin/python $(pwd)/app.py
Icon=$(pwd)/icon.png
Terminal=false
Type=Application
Categories=AudioVideo;Audio;Development;
EOF
    print_success "Desktop shortcut created"
fi

print_success "Installation complete! 🎉"
print_status "To start the application:"
print_status "  1. cd $(pwd)"
print_status "  2. source venv/bin/activate"
print_status "  3. python app.py"
print_status ""
print_status "The web interface will be available at http://localhost:7860"