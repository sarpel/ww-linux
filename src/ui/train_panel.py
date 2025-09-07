"""
Train Panel - ML training pipeline interface for wake word model training
"""

import gradio as gr
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import json
import time
from datetime import datetime
import random
import shutil
import os

# Import local modules
from ..utils.file_manager import PanelCommunicator

logger = logging.getLogger(__name__)

# Handle WakeWordTrainer import failure gracefully
try:
    from ..training.wake_word_trainer import WakeWordTrainer
    TRAINER_AVAILABLE = True
except Exception as e:
    logger.warning(f"⚠️ WakeWordTrainer not available: {e}")
    WakeWordTrainer = None
    TRAINER_AVAILABLE = False


class TrainPanel:
    """Train panel controller for ML model training and evaluation"""
    
    def __init__(self, communicator: PanelCommunicator):
        self.communicator = communicator
        if not TRAINER_AVAILABLE:
            logger.warning("⚠️ Training not available: WakeWordTrainer import failed")
            logger.warning("📋 Install CUDA Toolkit and TensorFlow for GPU training functionality")
            self.trainer = None
            self.training_available = False
        else:
            try:
                self.trainer = WakeWordTrainer()
                self.training_available = True
                logger.info("✅ GPU training available")
            except RuntimeError as e:
                logger.warning(f"⚠️ Training not available: {e}")
                logger.warning("📋 Install CUDA Toolkit for GPU training functionality")
                self.trainer = None
                self.training_available = False
        
        # Setup directories
        self.training_dir = Path("data/training")
        self.models_dir = Path("data/models")
        
        for dir_path in [self.training_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.training_active = False
        self.current_model = None
        self.training_history = []
    
    def _check_training_available(self) -> bool:
        """Check if training is available and return appropriate message if not"""
        if not self.training_available:
            return False
        return True
    
    def validate_training_data(self) -> Tuple[str, Dict]:
        """
        Validate training data structure and quality
        
        Returns:
            Tuple[str, Dict]: Status message and validation results
        """
        try:
            results = self.trainer.validate_dataset(self.training_dir)
            
            if results.get('valid', False):
                status = f"✅ Dataset validated: {results['total_samples']} samples"
                status += f" ({results['positive_samples']} positive, {results['negative_samples']} negative)"
                
                balance_ratio = results.get('balance_ratio', 0)
                if balance_ratio < 0.3 or balance_ratio > 0.7:
                    status += " ⚠️ Dataset imbalanced"
            else:
                status = f"❌ Dataset validation failed: {results.get('error', 'Unknown error')}"
            
            return status, results
            
        except Exception as e:
            logger.error(f"Error validating training data: {e}")
            return "<div class=\'error-panel\'>❌ Error: {str(e)}</div>", {}
    
    def configure_training(
        self,
        model_architecture: str,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        validation_split: float,
        early_stopping: bool,
        advanced_config: str
    ) -> Tuple[str, Dict]:
        """
        Configure training parameters
        
        Args:
            model_architecture: Model type ('cnn', 'rnn', 'transformer')
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            epochs: Maximum number of epochs
            validation_split: Fraction of data for validation
            early_stopping: Whether to use early stopping
            advanced_config: JSON string with advanced configuration
            
        Returns:
            Tuple[str, Dict]: Status message and configuration
        """
        try:
            # Parse advanced configuration
            advanced_params = {}
            if advanced_config.strip():
                try:
                    advanced_params = json.loads(advanced_config)
                except json.JSONDecodeError as e:
                    return "<div class=\'error-panel\'>❌ Error: Invalid advanced configuration JSON: {str(e)}</div>", {}
            
            # Build configuration
            config = {
                'model_architecture': model_architecture,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs,
                'validation_split': validation_split,
                'early_stopping': early_stopping,
                'advanced': advanced_params
            }
            
            # Validate configuration
            validation_result = self.trainer.validate_config(config)
            
            if validation_result.get('valid', False):
                self.trainer.set_config(config)
                status = f"✅ Training configuration set for {model_architecture.upper()} model"
                
                # Add configuration summary
                status += f" (LR: {learning_rate}, Batch: {batch_size}, Epochs: {epochs})"
            else:
                status = f"❌ Configuration validation failed: {validation_result.get('error', 'Unknown error')}"
            
            return status, config
            
        except Exception as e:
            logger.error(f"Error configuring training: {e}")
            return "<div class=\'error-panel\'>❌ Error: {str(e)}</div>", {}
    
    def start_training(self, progress=gr.Progress()) -> Tuple[str, str]:
        """
        Start training process
        
        Args:
            progress: Gradio progress tracker
            
        Returns:
            Tuple[str, str]: Status message and training info
        """
        try:
            if not self._check_training_available():
                return ("<div class='error-panel'>❌ GPU Training Not Available<br>"
                       "Install CUDA Toolkit for training functionality</div>", 
                       "Training unavailable - GPU required")
            
            if self.training_active:
                return "<div class=\'warning-panel\'>⚠️ Training is already active</div>", ""
            
            progress(0, desc="Initializing training...")
            
            # Validate data and configuration
            data_validation = self.trainer.validate_dataset(self.training_dir)
            if not data_validation.get('valid', False):
                return "<div class=\'error-panel\'>❌ Cannot start training: {data_validation.get('error', 'Invalid dataset')}</div>", ""
            
            config_validation = self.trainer.validate_current_config()
            if not config_validation.get('valid', False):
                return "<div class=\'error-panel\'>❌ Cannot start training: {config_validation.get('error', 'Invalid configuration')}</div>", ""
            
            progress(0.1, desc="Starting training process...")
            
            # Start training in background
            self.training_active = True
            training_info = {
                'start_time': datetime.now().isoformat(),
                'model_architecture': self.trainer.config.get('model_architecture', 'unknown'),
                'total_samples': data_validation.get('total_samples', 0),
                'status': 'running'
            }
            
            # Create training callback for progress updates
            def training_callback(epoch, total_epochs, metrics):
                progress_val = 0.1 + 0.8 * (epoch / total_epochs)
                progress(progress_val, desc=f"Epoch {epoch}/{total_epochs} - Loss: {metrics.get('loss', 'N/A'):.4f}")
                
                # Store training history
                self.training_history.append({
                    'epoch': epoch,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics
                })
            
            # Start training
            success = self.trainer.start_training(
                self.training_dir,
                callback=training_callback
            )
            
            if success:
                progress(1.0, desc="Training completed successfully!")
                self.training_active = False
                
                # Get training results
                results = self.trainer.get_training_results()
                
                status = f"✅ Training completed in {results.get('duration', 'unknown')} seconds"
                status += f" (Best accuracy: {results.get('best_accuracy', 0):.3f})"
                
                training_info['status'] = 'completed'
                training_info['results'] = results
                
            else:
                self.training_active = False
                status = "❌ Training failed"
                training_info['status'] = 'failed'
            
            return status, json.dumps(training_info, indent=2)
            
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            self.training_active = False
            return "<div class=\'error-panel\'>❌ Error: {str(e)}</div>", ""
    
    def stop_training(self) -> str:
        """Stop active training"""
        try:
            if not self.training_active:
                return "<div class=\'warning-panel\'>⚠️ No active training to stop</div>"
            
            success = self.trainer.stop_training()
            
            if success:
                self.training_active = False
                return "<div class=\'success-panel\'>✅ Training stopped successfully</div>"
            else:
                return "<div class=\'error-panel\'>❌ Failed to stop training</div>"
                
        except Exception as e:
            logger.error(f"Error stopping training: {e}")
            return "<div class=\'error-panel\'>❌ Error: {str(e)}</div>"
    
    def get_training_metrics(self) -> Dict:
        """Get current training metrics and history"""
        try:
            if not self.training_history:
                return {
                    'epochs': [],
                    'loss': [],
                    'accuracy': [],
                    'val_loss': [],
                    'val_accuracy': [],
                    'current_epoch': 0,
                    'status': 'not_started'
                }
            
            # Extract metrics from history
            epochs = [h['epoch'] for h in self.training_history]
            loss = [h['metrics'].get('loss', 0) for h in self.training_history]
            accuracy = [h['metrics'].get('accuracy', 0) for h in self.training_history]
            val_loss = [h['metrics'].get('val_loss', 0) for h in self.training_history]
            val_accuracy = [h['metrics'].get('val_accuracy', 0) for h in self.training_history]
            
            return {
                'epochs': epochs,
                'loss': loss,
                'accuracy': accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'current_epoch': epochs[-1] if epochs else 0,
                'status': 'running' if self.training_active else 'completed'
            }
            
        except Exception as e:
            logger.error(f"Error getting training metrics: {e}")
            return {'error': str(e)}
    
    def create_metrics_plot(self) -> Optional[str]:
        """Create training metrics plot"""
        try:
            metrics = self.get_training_metrics()
            
            if 'error' in metrics or not metrics.get('epochs'):
                return None
            
            # Import plotting libraries
            import matplotlib.pyplot as plt
            import tempfile
            
            # Create subplot for metrics
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            epochs = metrics['epochs']
            
            # Plot loss
            ax1.plot(epochs, metrics['loss'], 'b-', label='Training Loss', linewidth=2)
            if metrics['val_loss']:
                ax1.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot accuracy
            ax2.plot(epochs, metrics['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
            if metrics['val_accuracy']:
                ax2.plot(epochs, metrics['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            plt.savefig(temp_file.name, dpi=150, bbox_inches='tight')
            plt.close()
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error creating metrics plot: {e}")
            return None
    
    def evaluate_model(self, test_data_path: str = None) -> Tuple[str, Dict]:
        """
        Evaluate trained model
        
        Args:
            test_data_path: Optional path to test data (uses validation split if None)
            
        Returns:
            Tuple[str, Dict]: Status message and evaluation results
        """
        try:
            if not self.trainer.has_trained_model():
                return "<div class=\'error-panel\'>❌ No trained model available for evaluation</div>", {}
            
            # Use validation data if no test path provided
            if not test_data_path:
                test_data_path = self.training_dir
            
            results = self.trainer.evaluate_model(test_data_path)
            
            if results.get('success', False):
                status = f"✅ Model evaluation completed"
                status += f" (Accuracy: {results.get('accuracy', 0):.3f}, Loss: {results.get('loss', 0):.3f})"
                
                # Add detailed metrics
                if 'confusion_matrix' in results:
                    status += f", Precision: {results.get('precision', 0):.3f}, Recall: {results.get('recall', 0):.3f}"
            else:
                status = f"❌ Model evaluation failed: {results.get('error', 'Unknown error')}"
            
            return status, results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return "<div class=\'error-panel\'>❌ Error: {str(e)}</div>", {}
    
    def export_model(
        self,
        model_name: str,
        model_format: str,
        include_preprocessing: bool
    ) -> str:
        """
        Export trained model
        
        Args:
            model_name: Name for exported model
            model_format: Export format ('onnx', 'tflite', 'pytorch', 'openwakeword')
            include_preprocessing: Whether to include preprocessing pipeline
            
        Returns:
            str: Status message
        """
        try:
            if not self.trainer.has_trained_model():
                return "<div class=\'error-panel\'>❌ No trained model available for export</div>"
            
            if not model_name.strip():
                return "<div class=\'error-panel\'>❌ Please provide a model name</div>"
            
            # Create export directory
            export_dir = self.models_dir / model_name
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Export model
            result = self.trainer.export_model(
                export_dir,
                model_format,
                include_preprocessing
            )
            
            if result.get('success', False):
                status = f"✅ Model exported to {export_dir}"
                status += f" (Format: {model_format.upper()}, Size: {result.get('file_size_mb', 0):.1f} MB)"
            else:
                status = f"❌ Model export failed: {result.get('error', 'Unknown error')}"
            
            return status
            
        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            return "<div class=\'error-panel\'>❌ Error: {str(e)}</div>"
    
    def get_training_stats(self) -> str:
        """Get training statistics and status"""
        try:
            # Get dataset stats
            data_stats = self.communicator.file_manager.get_folder_stats(self.training_dir)
            model_stats = self.communicator.file_manager.get_folder_stats(self.models_dir)
            
            # Get current training status
            metrics = self.get_training_metrics()
            
            status_icon = {
                'not_started': '⏳',
                'running': '🔄',
                'completed': '✅',
                'failed': '❌'
            }.get(metrics.get('status', 'not_started'), '❓')
            
            return f"""
            **Training Statistics:**
            
            📊 **Training Status:** {status_icon} {metrics.get('status', 'Not started').title()}
            
            📁 **Training Data:**
            - Total files: {data_stats.get('audio_files', 0)}
            - Data size: {data_stats.get('folder_size_mb', 0):.1f} MB
            - Last modified: {data_stats.get('last_modified') or 'Unknown'}
            
            🤖 **Models:**
            - Exported models: {model_stats.get('file_count', 0)}
            - Models size: {model_stats.get('folder_size_mb', 0):.1f} MB
            
            📈 **Training Progress:**
            - Current epoch: {metrics.get('current_epoch', 0)}
            - Best accuracy: {max(metrics.get('accuracy', [0])) if metrics.get('accuracy') else 0:.3f}
            - Training samples: {len(self.training_history)}
            """
            
        except Exception as e:
            logger.error(f"Error getting training stats: {e}")
            return "Error getting statistics"
    
    def clear_training_data(self) -> str:
        """Clear training data and history"""
        try:
            # Clear training history
            self.training_history = []
            
            # Clear training directory
            removed, total = self.communicator.file_manager.clean_directory(
                self.training_dir,
                file_patterns=["*.*"]
            )
            
            if removed > 0:
                return "<div class=\'success-panel\'>✅ Cleared {removed} training files and history</div>"
            else:
                return "No training data to clear"
                
        except Exception as e:
            logger.error(f"Error clearing training data: {e}")
            return "<div class=\'error-panel\'>❌ Error: {str(e)}</div>"
    
    def balance_dataset(
        self, 
        dataset_type: str, 
        target_count: int, 
        homogeneous: bool = True
    ) -> str:
        """
        Balance dataset by randomly sampling files while preserving category distribution
        
        Args:
            dataset_type: 'positive', 'negative', or 'both'
            target_count: Target number of samples to keep
            homogeneous: Whether to sample homogeneously across subdirectories
            
        Returns:
            str: Status message
        """
        try:
            # Define dataset paths
            train_pos = self.training_dir / 'train' / 'positive'
            train_neg = self.training_dir / 'train' / 'negative'
            val_pos = self.training_dir / 'validation' / 'positive'
            val_neg = self.training_dir / 'validation' / 'negative'
            
            # Ensure directories exist
            for dir_path in [train_pos, train_neg, val_pos, val_neg]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            total_removed = 0
            messages = []
            
            if dataset_type in ['positive', 'both']:
                removed_pos = self._balance_single_dataset(
                    [train_pos, val_pos], 
                    target_count if dataset_type == 'positive' else int(target_count * 0.8),
                    'positive',
                    homogeneous
                )
                total_removed += removed_pos
                if removed_pos > 0:
                    messages.append(f"Removed {removed_pos} positive samples")
            
            if dataset_type in ['negative', 'both']:
                neg_target = target_count if dataset_type == 'negative' else int(target_count * 1.5)
                removed_neg = self._balance_single_dataset(
                    [train_neg, val_neg], 
                    neg_target,
                    'negative',
                    homogeneous
                )
                total_removed += removed_neg
                if removed_neg > 0:
                    messages.append(f"Removed {removed_neg} negative samples")
            
            if total_removed > 0:
                status_msg = f"✅ Dataset balanced: {', '.join(messages)}"
                return f"<div class='success-panel'>{status_msg}</div>"
            else:
                return "<div class='info-panel'>ℹ️ Dataset already balanced to target size</div>"
                
        except Exception as e:
            logger.error(f"Error balancing dataset: {e}")
            return f"<div class='error-panel'>❌ Error balancing dataset: {str(e)}</div>"
    
    def _balance_single_dataset(
        self, 
        directories: List[Path], 
        target_count: int, 
        dataset_name: str,
        homogeneous: bool = True
    ) -> int:
        """
        Balance a single dataset (positive or negative) across directories
        
        Args:
            directories: List of directories to balance
            target_count: Target total number of files
            dataset_name: Name for logging
            homogeneous: Whether to sample homogeneously across subdirectories
            
        Returns:
            int: Number of files removed
        """
        try:
            # Collect all WAV files with their subdirectory info
            all_files = []
            subdirs = {}
            
            for directory in directories:
                if not directory.exists():
                    continue
                    
                for wav_file in directory.rglob('*.wav'):
                    # Get subdirectory path relative to main directory
                    rel_path = wav_file.relative_to(directory)
                    subdir = str(rel_path.parent) if rel_path.parent != Path('.') else 'root'
                    
                    all_files.append(wav_file)
                    
                    if subdir not in subdirs:
                        subdirs[subdir] = []
                    subdirs[subdir].append(wav_file)
            
            total_files = len(all_files)
            
            if total_files <= target_count:
                logger.info(f"{dataset_name}: {total_files} files, target {target_count} - no balancing needed")
                return 0
            
            # Determine files to keep
            if homogeneous and len(subdirs) > 1:
                files_to_keep = self._sample_homogeneously(subdirs, target_count)
            else:
                # Simple random sampling
                files_to_keep = random.sample(all_files, target_count)
            
            # Remove files not in the keep list
            files_to_remove = [f for f in all_files if f not in files_to_keep]
            
            removed_count = 0
            for file_path in files_to_remove:
                try:
                    file_path.unlink()  # Delete file
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Could not remove {file_path}: {e}")
            
            logger.info(f"Balanced {dataset_name}: kept {len(files_to_keep)}, removed {removed_count}")
            return removed_count
            
        except Exception as e:
            logger.error(f"Error balancing {dataset_name} dataset: {e}")
            return 0
    
    def _sample_homogeneously(self, subdirs: Dict[str, List[Path]], target_count: int) -> List[Path]:
        """
        Sample files homogeneously across subdirectories to preserve category distribution
        
        Args:
            subdirs: Dictionary mapping subdirectory names to file lists
            target_count: Total number of files to keep
            
        Returns:
            List[Path]: Files to keep
        """
        try:
            if not subdirs:
                return []
            
            # Calculate proportional sampling for each subdirectory
            total_files = sum(len(files) for files in subdirs.values())
            files_to_keep = []
            
            # First pass: proportional sampling
            remaining_target = target_count
            remaining_subdirs = dict(subdirs)
            
            for subdir, files in subdirs.items():
                if remaining_target <= 0:
                    break
                
                # Calculate proportional sample size
                proportion = len(files) / total_files
                sample_size = int(proportion * target_count)
                
                # Don't exceed available files in this subdirectory
                sample_size = min(sample_size, len(files), remaining_target)
                
                if sample_size > 0:
                    sampled = random.sample(files, sample_size)
                    files_to_keep.extend(sampled)
                    remaining_target -= sample_size
                    
                    # Remove sampled files from remaining pool
                    remaining_subdirs[subdir] = [f for f in files if f not in sampled]
            
            # Second pass: fill remaining slots from largest subdirectories
            while remaining_target > 0 and any(remaining_subdirs.values()):
                # Find subdirectory with most remaining files
                largest_subdir = max(
                    remaining_subdirs.items(), 
                    key=lambda x: len(x[1])
                )[0]
                
                if not remaining_subdirs[largest_subdir]:
                    del remaining_subdirs[largest_subdir]
                    continue
                
                # Take one more file from largest subdirectory
                selected_file = random.choice(remaining_subdirs[largest_subdir])
                files_to_keep.append(selected_file)
                remaining_subdirs[largest_subdir].remove(selected_file)
                remaining_target -= 1
            
            logger.info(f"Homogeneous sampling: {len(files_to_keep)} files from {len(subdirs)} subdirectories")
            return files_to_keep
            
        except Exception as e:
            logger.error(f"Error in homogeneous sampling: {e}")
            # Fallback to simple random sampling
            all_files = [f for files in subdirs.values() for f in files]
            return random.sample(all_files, min(target_count, len(all_files)))
    
    def auto_balance_dataset(self) -> str:
        """
        Automatically balance dataset to 1:2 positive:negative ratio
        
        Returns:
            str: Status message
        """
        try:
            # Get current dataset stats
            validation_result = self.trainer.validate_dataset(self.training_dir)
            
            if not validation_result.get('valid', False):
                return "<div class='error-panel'>❌ Cannot auto-balance: invalid dataset</div>"
            
            positive_count = validation_result.get('positive_samples', 0)
            negative_count = validation_result.get('negative_samples', 0)
            
            if positive_count == 0 or negative_count == 0:
                return "<div class='error-panel'>❌ Cannot auto-balance: missing positive or negative samples</div>"
            
            # Calculate optimal balance (aim for 1:10 ratio, positive:negative)
            # Wake words are rare events, need strong negative discrimination
            target_negative = min(positive_count * 10, negative_count)
            target_positive = positive_count  # Keep all positive samples
            
            # Balance negative samples to target
            if negative_count > target_negative:
                result = self._balance_single_dataset(
                    [self.training_dir / 'train' / 'negative', self.training_dir / 'validation' / 'negative'],
                    target_negative,
                    'negative',
                    homogeneous=True
                )
                
                new_ratio = target_positive / (target_positive + target_negative) * 100
                return f"<div class='success-panel'>✅ Auto-balanced: {target_positive} positive, {target_negative} negative samples ({new_ratio:.1f}% positive)</div>"
            else:
                current_ratio = positive_count / (positive_count + negative_count) * 100
                return f"<div class='info-panel'>ℹ️ Dataset already well-balanced: {positive_count} positive, {negative_count} negative ({current_ratio:.1f}% positive)</div>"
                
        except Exception as e:
            logger.error(f"Error in auto-balance: {e}")
            return f"<div class='error-panel'>❌ Error auto-balancing dataset: {str(e)}</div>"


def create_train_panel(communicator: PanelCommunicator):
    """Create the train panel UI components"""
    
    panel = TrainPanel(communicator)
    
    with gr.Column(elem_classes=["panel-container"]):
        
        # Dataset validation section
        with gr.Group():
            gr.Markdown("### 📊 Dataset Validation")
            
            validate_btn = gr.Button("🔍 Validate Training Data", variant="secondary")
            validation_status = gr.HTML(value="Click to validate your training dataset")
        
        # Dataset balancing section
        with gr.Group():
            gr.Markdown("### ⚖️ Dataset Balancing")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Positive Samples**")
                    positive_target = gr.Number(
                        label="Target Count",
                        value=20000,
                        minimum=100,
                        step=100,
                        info="Number of positive samples to keep"
                    )
                    balance_positive_btn = gr.Button("🎯 Balance Positive", variant="secondary", size="sm")
                
                with gr.Column():
                    gr.Markdown("**Negative Samples**")
                    negative_target = gr.Number(
                        label="Target Count", 
                        value=40000,
                        minimum=100,
                        step=100,
                        info="Number of negative samples to keep"
                    )
                    balance_negative_btn = gr.Button("🎯 Balance Negative", variant="secondary", size="sm")
            
            with gr.Row():
                balance_both_btn = gr.Button("⚖️ Balance Both Datasets", variant="primary")
                auto_balance_btn = gr.Button("🔄 Auto-Balance (1:10 ratio)", variant="secondary")
            
            balancing_status = gr.HTML(value="Adjust target counts and click balance buttons")
        
        # Training configuration section
        with gr.Group():
            gr.Markdown("### ⚙️ Training Configuration")
            
            with gr.Row():
                model_architecture = gr.Dropdown(
                    label="Model Architecture",
                    choices=["cnn", "rnn", "transformer"],
                    value="cnn"
                )
                
                learning_rate = gr.Number(
                    label="Learning Rate",
                    value=0.001,
                    minimum=0.0001,
                    maximum=0.1,
                    step=0.0001
                )
            
            with gr.Row():
                batch_size = gr.Slider(
                    label="Batch Size",
                    minimum=8,
                    maximum=128,
                    value=32,
                    step=8
                )
                
                epochs = gr.Slider(
                    label="Max Epochs",
                    minimum=10,
                    maximum=500,
                    value=100,
                    step=10
                )
            
            with gr.Row():
                validation_split = gr.Slider(
                    label="Validation Split",
                    minimum=0.1,
                    maximum=0.4,
                    value=0.2,
                    step=0.05
                )
                
                early_stopping = gr.Checkbox(
                    label="Early Stopping",
                    value=True
                )
            
            advanced_config = gr.Code(
                label="Advanced Configuration (JSON)",
                language="json",
                value="""{
  "optimizer": "adam",
  "loss_function": "binary_crossentropy",
  "metrics": ["accuracy", "precision", "recall"],
  "callbacks": {
    "reduce_lr": {"monitor": "val_loss", "patience": 5},
    "model_checkpoint": {"save_best_only": true}
  }
}"""
            )
            
            configure_btn = gr.Button("⚙️ Set Configuration", variant="secondary")
            config_status = gr.HTML(value="Configure training parameters above")
        
        # Training control section
        with gr.Group():
            gr.Markdown("### 🚀 Training Control")
            
            with gr.Row():
                start_training_btn = gr.Button("▶️ Start Training", variant="primary", size="lg")
                stop_training_btn = gr.Button("⏹️ Stop Training", variant="stop")
            
            training_status = gr.HTML(value="Ready to start training")
            training_info = gr.Code(language="json", label="Training Information")
        
        # Metrics visualization section
        with gr.Group():
            gr.Markdown("### 📈 Training Metrics")
            
            with gr.Row():
                refresh_metrics_btn = gr.Button("🔄 Refresh Metrics", variant="secondary")
                metrics_plot = gr.Image(label="Training Progress", interactive=False)
            
            # Real-time metrics display
            metrics_display = gr.JSON(
                label="Current Metrics",
                value=panel.get_training_metrics()
            )
        
        # Model evaluation section
        with gr.Group():
            gr.Markdown("### 🎯 Model Evaluation")
            
            evaluate_btn = gr.Button("📊 Evaluate Model", variant="secondary")
            evaluation_status = gr.HTML(value="Train a model first to evaluate")
            
            evaluation_results = gr.JSON(
                label="Evaluation Results",
                value={}
            )
        
        # Model export section
        with gr.Group():
            gr.Markdown("### 💾 Model Export")
            
            with gr.Row():
                model_name = gr.Textbox(
                    label="Model Name",
                    placeholder="my_wake_word_model",
                    value="wake_word_model_v1"
                )
                
                model_format = gr.Dropdown(
                    label="Export Format",
                    choices=["onnx", "tflite", "pytorch", "openwakeword"],
                    value="onnx"
                )
            
            include_preprocessing = gr.Checkbox(
                label="Include Preprocessing Pipeline",
                value=True
            )
            
            export_btn = gr.Button("💾 Export Model", variant="primary")
            export_status = gr.HTML(value="Complete training to export model")
        
        # Statistics and management
        with gr.Group():
            gr.Markdown("### 📊 Statistics & Management")
            
            stats_display = gr.HTML(
                value=panel.get_training_stats()
            )
            
            with gr.Row():
                refresh_stats_btn = gr.Button("🔄 Refresh Stats", variant="secondary")
                clear_training_btn = gr.Button("🗑️ Clear Training Data", variant="stop")
        
        # Event handlers
        def on_validate_data():
            return panel.validate_training_data()[0]
        
        def on_configure_training(arch, lr, batch, eps, val_split, early_stop, advanced):
            return panel.configure_training(
                arch, lr, int(batch), int(eps), val_split, early_stop, advanced
            )[0]
        
        def on_start_training(progress=gr.Progress()):
            status, info = panel.start_training(progress)
            return status, info, panel.get_training_metrics()
        
        def on_stop_training():
            return panel.stop_training()
        
        def on_refresh_metrics():
            metrics = panel.get_training_metrics()
            plot = panel.create_metrics_plot()
            return metrics, plot
        
        def on_evaluate_model():
            status, results = panel.evaluate_model()
            return status, results
        
        def on_export_model(name, fmt, include_prep):
            return panel.export_model(name, fmt, include_prep)
        
        def on_refresh_stats():
            return panel.get_training_stats()
        
        def on_clear_training():
            return panel.clear_training_data()
        
        def on_balance_positive(target_count):
            return panel.balance_dataset('positive', int(target_count))
        
        def on_balance_negative(target_count):
            return panel.balance_dataset('negative', int(target_count))
        
        def on_balance_both(pos_target, neg_target):
            pos_result = panel.balance_dataset('positive', int(pos_target))
            neg_result = panel.balance_dataset('negative', int(neg_target))
            return f"{pos_result}<br>{neg_result}"
        
        def on_auto_balance():
            return panel.auto_balance_dataset()
        
        # Connect events
        validate_btn.click(
            fn=on_validate_data,
            outputs=[validation_status]
        )
        
        balance_positive_btn.click(
            fn=on_balance_positive,
            inputs=[positive_target],
            outputs=[balancing_status]
        )
        
        balance_negative_btn.click(
            fn=on_balance_negative,
            inputs=[negative_target],
            outputs=[balancing_status]
        )
        
        balance_both_btn.click(
            fn=on_balance_both,
            inputs=[positive_target, negative_target],
            outputs=[balancing_status]
        )
        
        auto_balance_btn.click(
            fn=on_auto_balance,
            outputs=[balancing_status]
        )
        
        configure_btn.click(
            fn=on_configure_training,
            inputs=[model_architecture, learning_rate, batch_size, epochs, 
                   validation_split, early_stopping, advanced_config],
            outputs=[config_status]
        )
        
        start_training_btn.click(
            fn=on_start_training,
            outputs=[training_status, training_info, metrics_display]
        )
        
        stop_training_btn.click(
            fn=on_stop_training,
            outputs=[training_status]
        )
        
        refresh_metrics_btn.click(
            fn=on_refresh_metrics,
            outputs=[metrics_display, metrics_plot]
        )
        
        evaluate_btn.click(
            fn=on_evaluate_model,
            outputs=[evaluation_status, evaluation_results]
        )
        
        export_btn.click(
            fn=on_export_model,
            inputs=[model_name, model_format, include_preprocessing],
            outputs=[export_status]
        )
        
        refresh_stats_btn.click(
            fn=on_refresh_stats,
            outputs=[stats_display]
        )
        
        clear_training_btn.click(
            fn=on_clear_training,
            outputs=[training_status]
        )
    
    return {
        'validate_btn': validate_btn,
        'validation_status': validation_status,
        'positive_target': positive_target,
        'negative_target': negative_target,
        'balance_positive_btn': balance_positive_btn,
        'balance_negative_btn': balance_negative_btn,
        'balance_both_btn': balance_both_btn,
        'auto_balance_btn': auto_balance_btn,
        'balancing_status': balancing_status,
        'model_architecture': model_architecture,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'validation_split': validation_split,
        'early_stopping': early_stopping,
        'advanced_config': advanced_config,
        'configure_btn': configure_btn,
        'config_status': config_status,
        'start_training_btn': start_training_btn,
        'stop_training_btn': stop_training_btn,
        'training_status': training_status,
        'training_info': training_info,
        'metrics_display': metrics_display,
        'metrics_plot': metrics_plot,
        'evaluate_btn': evaluate_btn,
        'evaluation_status': evaluation_status,
        'evaluation_results': evaluation_results,
        'model_name': model_name,
        'model_format': model_format,
        'include_preprocessing': include_preprocessing,
        'export_btn': export_btn,
        'export_status': export_status,
        'stats_display': stats_display
    }