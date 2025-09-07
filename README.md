# 🐧 Wake Word Generator - Linux Ubuntu Edition

**Optimized for Ubuntu 20.04+ with full GPU training support**

This is the Linux-optimized version of the Wake Word Generator, designed specifically for Ubuntu systems with NVIDIA GPU acceleration.

## 🚀 Features

- **Full GPU Training**: Multi-GPU support with CUDA acceleration
- **Linux Optimizations**: Memory-mapped datasets, optimized I/O operations
- **Advanced Training**: Mixed precision training, automatic model export to ONNX
- **Real-time Monitoring**: GPU utilization, training metrics, system status
- **Production Ready**: Containerization support, logging, monitoring

## 📋 Requirements

### System Requirements
- **OS**: Ubuntu 20.04+ (or compatible Linux distribution)
- **GPU**: NVIDIA GPU with CUDA Compute Capability 6.0+
- **CUDA**: CUDA 11.8+ (recommended)
- **Python**: Python 3.8+
- **Memory**: 8GB+ RAM, 4GB+ GPU VRAM recommended

### Hardware Recommendations
- **Training**: RTX 3060 Ti or better (8GB+ VRAM)
- **Production**: GTX 1660 or better (6GB+ VRAM)
- **CPU**: 8+ cores recommended for data loading
- **Storage**: SSD recommended for dataset I/O

## 🔧 Installation

### Quick Installation (Recommended)
```bash
# Clone or extract to desired location
cd wakeword-linux

# Run automated installation script
chmod +x install.sh
./install.sh

# Activate virtual environment
source venv/bin/activate

# Start the application
python app.py
```

### Manual Installation
```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3-dev python3-pip python3-venv build-essential \
    cmake pkg-config libasound2-dev portaudio19-dev libsndfile1-dev \
    libfftw3-dev libatlas-base-dev gfortran libhdf5-dev git curl wget \
    unzip ffmpeg sox

# Install NVIDIA drivers (if not already installed)
sudo ubuntu-drivers autoinstall

# Install CUDA Toolkit (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow[and-cuda]==2.15.0
pip install -r requirements.txt
```

## 🚀 Usage

### Starting the Application
```bash
# Activate virtual environment
source venv/bin/activate

# Start the web interface
python app.py
```

The web interface will be available at `http://localhost:7860`

### Training Workflow

1. **Generate**: Create wake word audio samples using Piper TTS or ElevenLabs
2. **Augment**: Upload and process training data with background noise
3. **GPU Train**: Train models with full GPU acceleration

### GPU Training Features

- **Multi-GPU Support**: Automatic detection and usage of multiple GPUs
- **Mixed Precision**: Automatic mixed precision training for faster performance
- **Memory Optimization**: Efficient memory usage with gradient checkpointing
- **Real-time Monitoring**: Live GPU utilization and training metrics
- **Model Export**: Automatic ONNX export for deployment

### Command Line Interface
```bash
# Check GPU status
python -c "
import torch
print('CUDA Available:', torch.cuda.is_available())
print('GPU Count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU Name:', torch.cuda.get_device_name(0))
"

# Validate training data
python -c "
from src.training.wake_word_trainer_linux import LinuxWakeWordTrainer
from pathlib import Path
trainer = LinuxWakeWordTrainer()
results = trainer.validate_dataset(Path('data/training'))
print('Dataset Valid:', results['valid'])
print('Wake word files:', results['wake_word_files'])
print('Background files:', results['background_files'])
"
```

## 📊 Performance Optimization

### GPU Memory Settings
```bash
# Set GPU memory growth (prevents OOM)
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Set PyTorch memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Training Optimization
- **Batch Size**: Start with 32, increase if you have more VRAM
- **Workers**: Set to number of CPU cores for optimal data loading
- **Mixed Precision**: Enabled by default for RTX 20 series and newer
- **Multi-GPU**: Automatically enabled if multiple GPUs detected

### Dataset Recommendations
- **Wake Word Samples**: 100-500 high-quality samples
- **Background Samples**: 1000+ diverse background/negative samples
- **Audio Format**: 16kHz WAV files for optimal performance
- **Duration**: 1-3 seconds per sample

## 🐳 Docker Deployment

### Build Docker Image
```bash
# Build with GPU support
docker build -t wakeword-generator-linux .

# Run with GPU access
docker run --gpus all -p 7860:7860 wakeword-generator-linux
```

### Docker Compose
```yaml
version: '3.8'
services:
  wakeword-generator:
    build: .
    ports:
      - "7860:7860"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./data:/app/data
      - ./models:/app/models
```

## 📝 Configuration

### Training Configuration
Edit the configuration in `src/training/wake_word_trainer_linux.py`:

```python
config = {
    'sample_rate': 16000,
    'n_mels': 80,
    'batch_size': 32,  # Increase for more VRAM
    'epochs': 50,
    'learning_rate': 0.001,
    'mixed_precision': True,  # Enable for RTX 20+
    'multi_gpu': True,       # Enable multi-GPU
    'num_workers': 8,        # Set to CPU count
}
```

### Environment Variables
```bash
# GPU Selection
export CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1

# Performance
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Logging
export PYTHONUNBUFFERED=1
```

## 🔧 Troubleshooting

### CUDA Issues
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Test PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Test TensorFlow GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Memory Issues
- Reduce batch size in configuration
- Enable gradient checkpointing
- Close other GPU applications
- Use `nvidia-smi` to monitor GPU memory

### Performance Issues
- Check GPU utilization with `nvidia-smi`
- Increase number of data loading workers
- Use SSD storage for datasets
- Enable mixed precision training

## 📈 Monitoring

### System Monitoring
```bash
# GPU monitoring
watch -n 1 nvidia-smi

# System resources
htop

# Disk I/O
iotop
```

### Application Logs
```bash
# View application logs
tail -f wakeword-linux.log

# Training progress
grep "Epoch" wakeword-linux.log
```

## 🚀 Production Deployment

### Systemd Service
Create `/etc/systemd/system/wakeword-generator.service`:

```ini
[Unit]
Description=Wake Word Generator Linux
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/wakeword-linux
Environment=PATH=/opt/wakeword-linux/venv/bin
ExecStart=/opt/wakeword-linux/venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable wakeword-generator
sudo systemctl start wakeword-generator
```

### Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 📋 License

Same as the main project. See LICENSE file in the parent directory.

## 🤝 Contributing

This Linux version maintains compatibility with the main Windows version while adding Linux-specific optimizations. Contributions welcome!

## 📞 Support

- Check the main project documentation
- GPU issues: Ensure CUDA and drivers are properly installed
- Performance: Monitor system resources and GPU utilization
- Training: Validate datasets and check logs for detailed error information