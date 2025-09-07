"""
Linux-optimized Wake Word Trainer with full GPU support
Designed for Ubuntu 20.04+ with CUDA 11.8+
"""

import os
import warnings
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import psutil
import subprocess

# Suppress warnings before imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T

import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import librosa

logger = logging.getLogger(__name__)


class LinuxWakeWordTrainer:
    """
    Linux-optimized wake word trainer with full GPU support
    Features:
    - Multi-GPU training support
    - Memory-mapped dataset loading
    - Advanced data augmentation
    - Real-time training monitoring
    - Automatic mixed precision training
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Initialize GPU environment
        self._setup_gpu_environment()
        
        # Setup device
        self.device = self._select_best_device()
        
        # Initialize training state
        self.model = None
        self.training_active = False
        self.training_history = []
        
        logger.info(f"🔥 Linux Wake Word Trainer initialized on {self.device}")
    
    def _default_config(self) -> Dict:
        """Default configuration optimized for Linux"""
        return {
            'sample_rate': 16000,
            'n_mels': 80,
            'n_fft': 512,
            'hop_length': 160,
            'win_length': 400,
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001,
            'mixed_precision': True,
            'multi_gpu': True,
            'num_workers': psutil.cpu_count(),
            'pin_memory': True,
        }
    
    def _setup_gpu_environment(self):
        """Setup optimal GPU environment for Linux"""
        # CUDA settings
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            
            # Set memory allocation strategy
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        # TensorFlow GPU settings
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth to prevent taking all GPU memory
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set GPU as logical device
                tf.config.set_visible_devices(gpus, 'GPU')
                
                logger.info(f"🚀 TensorFlow GPU setup complete: {len(gpus)} GPUs")
            except RuntimeError as e:
                logger.warning(f"GPU memory configuration failed: {e}")
    
    def _select_best_device(self) -> torch.device:
        """Select the best available device with detailed info"""
        if torch.cuda.is_available():
            # Get GPU information
            gpu_count = torch.cuda.device_count()
            device = torch.device('cuda:0')
            
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            logger.info(f"🔥 GPU Training Mode: {gpu_name}")
            logger.info(f"📊 GPU Memory: {gpu_memory:.1f}GB")
            logger.info(f"🔢 GPU Count: {gpu_count}")
            
            if gpu_count > 1 and self.config['multi_gpu']:
                logger.info("🚀 Multi-GPU training enabled")
            
            return device
        else:
            logger.warning("⚠️ CUDA not available, falling back to CPU")
            logger.warning("📋 Install NVIDIA drivers and CUDA toolkit for GPU training")
            return torch.device('cpu')
    
    def get_system_info(self) -> Dict:
        """Get detailed system information"""
        info = {
            'platform': 'Linux Ubuntu',
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            'pytorch_version': torch.__version__,
            'tensorflow_version': tf.__version__,
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / 1e9,
            'gpu_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
            })
        
        return info
    
    def validate_dataset(self, data_path: Path) -> Dict:
        """
        Comprehensive dataset validation with Linux optimizations
        """
        results = {
            'valid': False,
            'wake_word_files': 0,
            'background_files': 0,
            'total_duration': 0.0,
            'sample_rates': set(),
            'issues': [],
            'recommendations': []
        }
        
        try:
            data_path = Path(data_path)
            
            # Check positive samples
            positive_dir = data_path / 'positive'
            negative_dir = data_path / 'negative' 
            background_dir = data_path / 'background'
            
            if not positive_dir.exists():
                results['issues'].append("Missing 'positive' directory")
                return results
            
            # Analyze positive samples
            positive_files = list(positive_dir.glob('*.wav'))
            results['wake_word_files'] = len(positive_files)
            
            # Analyze negative/background samples
            negative_files = []
            if negative_dir.exists():
                negative_files.extend(list(negative_dir.glob('*.wav')))
            if background_dir.exists():
                negative_files.extend(list(background_dir.glob('*.wav')))
            
            results['background_files'] = len(negative_files)
            
            # Sample a few files for analysis (Linux I/O optimized)
            sample_files = positive_files[:10] + negative_files[:10]
            
            for audio_file in sample_files:
                try:
                    # Use librosa for consistent loading
                    y, sr = librosa.load(audio_file, sr=None)
                    results['sample_rates'].add(sr)
                    results['total_duration'] += len(y) / sr
                except Exception as e:
                    results['issues'].append(f"Error loading {audio_file.name}: {e}")
            
            # Validation checks
            if results['wake_word_files'] < 10:
                results['issues'].append(f"Need at least 10 positive samples (found {results['wake_word_files']})")
            
            if results['background_files'] < 50:
                results['recommendations'].append(f"Recommend at least 50 negative samples (found {results['background_files']})")
            
            # Check sample rate consistency
            if len(results['sample_rates']) > 1:
                results['issues'].append(f"Inconsistent sample rates: {results['sample_rates']}")
                results['recommendations'].append("Resample all audio to 16kHz for optimal training")
            
            results['valid'] = len(results['issues']) == 0
            results['sample_rates'] = list(results['sample_rates'])
            
            return results
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            results['issues'].append(f"Validation error: {e}")
            return results


class WakeWordDataset(Dataset):
    """
    Linux-optimized dataset with memory mapping and advanced augmentation
    """
    
    def __init__(self, data_path: Path, config: Dict, augment: bool = True):
        self.data_path = Path(data_path)
        self.config = config
        self.augment = augment
        
        # Audio transforms
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config['sample_rate'],
            n_mels=config['n_mels'],
            n_fft=config['n_fft'],
            hop_length=config['hop_length'],
            win_length=config['win_length']
        )
        
        # Augmentation transforms (Linux optimized)
        if augment:
            self.time_shift = T.TimeShift(shift=0.1)
            self.pitch_shift = T.PitchShift(config['sample_rate'], n_steps=2)
            self.add_noise = lambda x: x + 0.005 * torch.randn_like(x)
        
        # Load file paths and labels
        self._load_file_paths()
    
    def _load_file_paths(self):
        """Load and index all audio files"""
        self.file_paths = []
        self.labels = []
        
        # Positive samples (wake word)
        positive_dir = self.data_path / 'positive'
        if positive_dir.exists():
            for wav_file in positive_dir.glob('*.wav'):
                self.file_paths.append(wav_file)
                self.labels.append(1)
        
        # Negative samples
        for neg_dir_name in ['negative', 'background']:
            neg_dir = self.data_path / neg_dir_name
            if neg_dir.exists():
                for wav_file in neg_dir.glob('*.wav'):
                    self.file_paths.append(wav_file)
                    self.labels.append(0)
        
        logger.info(f"📊 Dataset loaded: {len(self.file_paths)} files")
        logger.info(f"   Positive: {sum(self.labels)} | Negative: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """Load and process audio with augmentation"""
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Ensure mono and correct sample rate
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if sample_rate != self.config['sample_rate']:
                resampler = T.Resample(sample_rate, self.config['sample_rate'])
                waveform = resampler(waveform)
            
            # Apply augmentation
            if self.augment and torch.rand(1) > 0.5:
                # Random augmentation
                aug_choice = torch.randint(0, 3, (1,)).item()
                if aug_choice == 0:
                    waveform = self.time_shift(waveform)
                elif aug_choice == 1:
                    waveform = self.pitch_shift(waveform)
                else:
                    waveform = self.add_noise(waveform)
            
            # Convert to mel spectrogram
            mel_spec = self.mel_transform(waveform)
            mel_spec = torch.log(mel_spec + 1e-9)  # Log scale
            
            return mel_spec.squeeze(0), torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            # Return zero tensor on error
            zero_spec = torch.zeros(self.config['n_mels'], 100)
            return zero_spec, torch.tensor(label, dtype=torch.long)


class WakeWordModel(nn.Module):
    """
    Optimized CNN model for wake word detection
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
        )
        
        # Calculate feature size dynamically
        self._calculate_feature_size(config)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # Binary classification
        )
    
    def _calculate_feature_size(self, config):
        """Calculate the size of features after conv layers"""
        with torch.no_grad():
            # Dummy input
            x = torch.zeros(1, 1, config['n_mels'], 100)
            x = self.conv_layers(x)
            self.feature_size = x.view(1, -1).size(1)
    
    def forward(self, x):
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


    def train_model(self, data_path: Path, progress_callback=None) -> Dict:
        """
        Train wake word model with Linux optimizations
        """
        try:
            # Validate dataset first
            validation_results = self.validate_dataset(data_path)
            if not validation_results['valid']:
                return {
                    'success': False,
                    'error': 'Dataset validation failed',
                    'details': validation_results['issues']
                }
            
            # Create datasets
            dataset = WakeWordDataset(data_path, self.config, augment=True)
            
            # Split dataset
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            # Create data loaders with Linux optimizations
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['num_workers'],
                pin_memory=self.config['pin_memory'],
                persistent_workers=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                pin_memory=self.config['pin_memory'],
                persistent_workers=True
            )
            
            # Initialize model
            self.model = WakeWordModel(self.config).to(self.device)
            
            # Multi-GPU setup
            if torch.cuda.device_count() > 1 and self.config['multi_gpu']:
                self.model = nn.DataParallel(self.model)
                logger.info(f"🚀 Using {torch.cuda.device_count()} GPUs for training")
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            
            # Mixed precision training
            scaler = torch.cuda.amp.GradScaler() if self.config['mixed_precision'] else None
            
            # Training loop
            self.training_active = True
            best_val_acc = 0.0
            training_history = []
            
            for epoch in range(self.config['epochs']):
                if not self.training_active:
                    break
                
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (data, targets) in enumerate(train_loader):
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    if scaler:  # Mixed precision
                        with torch.cuda.amp.autocast():
                            outputs = self.model(data)
                            loss = criterion(outputs, targets)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = self.model(data)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += targets.size(0)
                    train_correct += predicted.eq(targets).sum().item()
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for data, targets in val_loader:
                        data, targets = data.to(self.device), targets.to(self.device)
                        outputs = self.model(data)
                        loss = criterion(outputs, targets)
                        
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()
                
                # Calculate metrics
                train_acc = 100. * train_correct / train_total
                val_acc = 100. * val_correct / val_total
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                # Update scheduler
                scheduler.step(val_loss)
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_model(data_path.parent / 'models')
                
                # Record history
                epoch_info = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'lr': optimizer.param_groups[0]['lr']
                }
                training_history.append(epoch_info)
                
                # Progress callback
                if progress_callback:
                    progress_callback(epoch_info)
                
                logger.info(
                    f"Epoch {epoch+1}/{self.config['epochs']}: "
                    f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}"
                )
            
            self.training_active = False
            self.training_history = training_history
            
            return {
                'success': True,
                'best_accuracy': best_val_acc,
                'epochs_trained': len(training_history),
                'final_loss': training_history[-1]['val_loss'] if training_history else 0,
                'training_history': training_history
            }
            
        except Exception as e:
            self.training_active = False
            logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'training_history': self.training_history
            }
    
    def _save_model(self, models_dir: Path):
        """Save model with Linux-specific optimizations"""
        models_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        model_path = models_dir / f"wake_word_model_{timestamp}.pth"
        
        # Save model state
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'timestamp': timestamp,
            'platform': 'Linux Ubuntu'
        }
        
        torch.save(save_dict, model_path)
        
        # Create ONNX export for deployment
        try:
            self.export_onnx(model_path.with_suffix('.onnx'))
            logger.info(f"✅ Model saved: {model_path}")
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")
    
    def export_onnx(self, output_path: Path):
        """Export model to ONNX format for deployment"""
        if self.model is None:
            raise RuntimeError("No trained model available for export")
        
        self.model.eval()
        
        # Dummy input
        dummy_input = torch.randn(1, self.config['n_mels'], 100).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            input_names=['audio_features'],
            output_names=['predictions'],
            dynamic_axes={
                'audio_features': {2: 'time_steps'},
                'predictions': {0: 'batch_size'}
            }
        )
        
        logger.info(f"📦 ONNX model exported: {output_path}")
    
    def stop_training(self):
        """Stop training gracefully"""
        self.training_active = False
        logger.info("🛑 Training stopped by user")


# Export main class for compatibility
WakeWordTrainer = LinuxWakeWordTrainer