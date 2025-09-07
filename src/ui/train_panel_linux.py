"""
Linux-optimized Train Panel with full GPU training support
"""

import gradio as gr
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import json
import time
from datetime import datetime
import threading
import psutil
import os

# Import Linux-optimized trainer
from ..training.wake_word_trainer_linux import LinuxWakeWordTrainer
from ..utils.file_manager import PanelCommunicator

logger = logging.getLogger(__name__)


class LinuxTrainPanel:
    """Linux-optimized train panel with GPU acceleration"""
    
    def __init__(self, communicator: PanelCommunicator):
        self.communicator = communicator
        
        # Try to initialize trainer
        try:
            self.trainer = LinuxWakeWordTrainer()
            self.training_available = True
            logger.info("🚀 Linux GPU training system initialized")
        except Exception as e:
            logger.error(f"❌ Training system failed to initialize: {e}")
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
        self.training_thread = None
    
    def get_system_status(self) -> str:
        """Get detailed Linux system status"""
        if not self.training_available:
            return "❌ Training system not available"
        
        try:
            sys_info = self.trainer.get_system_info()
            
            status_parts = [
                f"🐧 {sys_info.get('platform', 'Linux')}",
                f"🐍 Python {sys_info.get('python_version', 'Unknown')}",
                f"💾 {sys_info.get('memory_gb', 0):.1f}GB RAM",
                f"🔧 {sys_info.get('cpu_count', 0)} CPUs"
            ]
            
            if sys_info.get('gpu_available'):
                status_parts.extend([
                    f"🚀 {sys_info.get('gpu_name', 'GPU')}",
                    f"📊 {sys_info.get('gpu_memory_gb', 0):.1f}GB VRAM",
                    f"🔥 CUDA {sys_info.get('cuda_version', 'Unknown')}"
                ])
                
                if sys_info.get('gpu_count', 0) > 1:
                    status_parts.append(f"🔢 {sys_info['gpu_count']} GPUs")
            else:
                status_parts.append("⚠️ No GPU detected")
            
            return " | ".join(status_parts)
            
        except Exception as e:
            return f"❌ Status error: {e}"
    
    def validate_training_data(self) -> Tuple[str, Dict]:
        """Validate training data with Linux optimizations"""
        if not self.training_available:
            return "❌ Training system not available", {}
        
        try:
            results = self.trainer.validate_dataset(self.training_dir)
            
            if results['valid']:
                status_msg = f"""
                ✅ **Dataset Valid for Linux GPU Training**
                
                📊 **Statistics:**
                - Wake word samples: {results['wake_word_files']}
                - Background samples: {results['background_files']}
                - Total duration: {results['total_duration']:.1f}s
                - Sample rates: {results['sample_rates']}
                
                🚀 **Ready for GPU acceleration!**
                """
            else:
                issues = "\n".join([f"• {issue}" for issue in results['issues']])
                recommendations = "\n".join([f"• {rec}" for rec in results['recommendations']])
                
                status_msg = f"""
                ❌ **Dataset Issues Found**
                
                **Issues:**
                {issues}
                
                **Recommendations:**
                {recommendations}
                """
            
            return status_msg, results
            
        except Exception as e:
            error_msg = f"❌ Validation failed: {e}"
            logger.error(error_msg)
            return error_msg, {}
    
    def start_training(self, progress=gr.Progress()) -> Tuple[str, str, str]:
        """Start GPU-accelerated training with progress tracking"""
        if not self.training_available:
            return "❌ Training system not available", "", ""
        
        if self.training_active:
            return "⚠️ Training already in progress", "", ""
        
        try:
            # Validate dataset first
            validation_msg, validation_results = self.validate_training_data()
            if not validation_results.get('valid', False):
                return f"❌ Cannot start training:\n{validation_msg}", "", ""
            
            # Start training in background thread
            self.training_active = True
            self.training_thread = threading.Thread(
                target=self._training_worker,
                args=(progress,),
                daemon=True
            )
            self.training_thread.start()
            
            return "🚀 GPU training started! Monitor progress below...", "", ""
            
        except Exception as e:
            self.training_active = False
            error_msg = f"❌ Failed to start training: {e}"
            logger.error(error_msg)
            return error_msg, "", ""
    
    def _training_worker(self, progress):
        """Background training worker with progress updates"""
        def progress_callback(epoch_info):
            """Update progress during training"""
            epoch = epoch_info['epoch']
            total_epochs = self.trainer.config['epochs']
            
            progress_pct = epoch / total_epochs
            progress(progress_pct, f"Epoch {epoch}/{total_epochs} - Val Acc: {epoch_info['val_acc']:.2f}%")
            
            # Update history
            self.training_history.append(epoch_info)
        
        try:
            # Start training
            logger.info("🚀 Starting Linux GPU training...")
            results = self.trainer.train_model(
                self.training_dir,
                progress_callback=progress_callback
            )
            
            if results['success']:
                logger.info(f"✅ Training completed! Best accuracy: {results['best_accuracy']:.2f}%")
            else:
                logger.error(f"❌ Training failed: {results.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"💥 Training worker error: {e}")
        finally:
            self.training_active = False
    
    def stop_training(self) -> str:
        """Stop training gracefully"""
        if not self.training_active:
            return "⚠️ No training in progress"
        
        try:
            if self.trainer:
                self.trainer.stop_training()
            
            self.training_active = False
            
            # Wait for thread to finish
            if self.training_thread and self.training_thread.is_alive():
                self.training_thread.join(timeout=5.0)
            
            return "🛑 Training stopped successfully"
            
        except Exception as e:
            error_msg = f"❌ Error stopping training: {e}"
            logger.error(error_msg)
            return error_msg
    
    def get_training_status(self) -> Tuple[str, str, str]:
        """Get current training status with metrics"""
        try:
            if not self.training_available:
                return "❌ Training system not available", "", ""
            
            # Basic status
            if self.training_active:
                status = "🚀 GPU training in progress..."
            else:
                status = "⏸️ Training idle"
            
            # Training history
            history_text = ""
            if self.training_history:
                recent_epochs = self.training_history[-5:]  # Last 5 epochs
                history_text = "**Recent Training Progress:**\n\n"
                for epoch_info in recent_epochs:
                    history_text += (
                        f"Epoch {epoch_info['epoch']}: "
                        f"Train Acc {epoch_info['train_acc']:.2f}%, "
                        f"Val Acc {epoch_info['val_acc']:.2f}%, "
                        f"Loss {epoch_info['val_loss']:.4f}\n"
                    )
            
            # GPU utilization (if available)
            gpu_info = ""
            try:
                import torch
                if torch.cuda.is_available() and self.training_active:
                    gpu_util = torch.cuda.utilization()
                    gpu_memory = torch.cuda.memory_allocated() / 1e9
                    gpu_info = f"🔥 GPU Utilization: {gpu_util}% | Memory: {gpu_memory:.1f}GB"
            except:
                pass
            
            return status, history_text, gpu_info
            
        except Exception as e:
            error_msg = f"❌ Status error: {e}"
            return error_msg, "", ""
    
    def list_trained_models(self) -> List[Dict]:
        """List available trained models"""
        try:
            models = []
            
            if self.models_dir.exists():
                for model_file in self.models_dir.glob("*.pth"):
                    try:
                        # Get file info
                        stat = model_file.stat()
                        size_mb = stat.st_size / 1e6
                        modified = datetime.fromtimestamp(stat.st_mtime)
                        
                        models.append({
                            'name': model_file.name,
                            'path': str(model_file),
                            'size_mb': size_mb,
                            'modified': modified.strftime("%Y-%m-%d %H:%M:%S"),
                            'platform': 'Linux Ubuntu'
                        })
                    except Exception as e:
                        logger.warning(f"Error reading model {model_file}: {e}")
            
            # Sort by modification time (newest first)
            models.sort(key=lambda x: x['modified'], reverse=True)
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []


def create_train_panel_linux(communicator: PanelCommunicator):
    """Create Linux-optimized training panel interface"""
    
    panel = LinuxTrainPanel(communicator)
    
    with gr.Column() as train_panel:
        # System status
        with gr.Row():
            system_status = gr.HTML(
                value=f"<div class='linux-info-panel'>{panel.get_system_status()}</div>"
            )
        
        # Refresh system status
        refresh_system_btn = gr.Button("🔄 Refresh System Status", variant="secondary", size="sm")
        
        with gr.Tabs():
            # Data Validation Tab
            with gr.TabItem("📋 Data Validation"):
                gr.Markdown("### Validate training data for Linux GPU training")
                
                validate_btn = gr.Button("🔍 Validate Training Data", variant="primary")
                validation_output = gr.Markdown(value="Click 'Validate Training Data' to check your dataset")
                
                validate_btn.click(
                    fn=panel.validate_training_data,
                    outputs=[validation_output, gr.State()]  # Second output ignored
                )
            
            # Training Tab
            with gr.TabItem("🚀 GPU Training"):
                gr.Markdown("### Linux GPU-accelerated model training")
                
                with gr.Row():
                    start_train_btn = gr.Button("🚀 Start GPU Training", variant="primary")
                    stop_train_btn = gr.Button("🛑 Stop Training", variant="stop")
                
                training_output = gr.Markdown(value="Ready to start GPU training")
                
                # Training progress
                with gr.Group():
                    gr.Markdown("#### Training Progress")
                    progress_output = gr.Markdown(value="No training in progress")
                    gpu_output = gr.Markdown(value="")
                
                # Training actions
                start_train_btn.click(
                    fn=panel.start_training,
                    outputs=[training_output, progress_output, gpu_output]
                )
                
                stop_train_btn.click(
                    fn=panel.stop_training,
                    outputs=[training_output]
                )
            
            # Models Tab
            with gr.TabItem("📦 Trained Models"):
                gr.Markdown("### Manage trained models")
                
                refresh_models_btn = gr.Button("🔄 Refresh Models List", variant="secondary")
                
                def format_models_list():
                    models = panel.list_trained_models()
                    if not models:
                        return "No trained models found"
                    
                    models_text = "**Available Models:**\n\n"
                    for model in models:
                        models_text += (
                            f"• **{model['name']}**\n"
                            f"  Size: {model['size_mb']:.1f}MB | "
                            f"Modified: {model['modified']} | "
                            f"Platform: {model['platform']}\n\n"
                        )
                    
                    return models_text
                
                models_output = gr.Markdown(value=format_models_list())
                
                refresh_models_btn.click(
                    fn=lambda: format_models_list(),
                    outputs=[models_output]
                )
        
        # Auto-refresh training status
        def auto_refresh():
            """Auto-refresh training status"""
            if panel.training_active:
                return panel.get_training_status()
            return gr.skip(), gr.skip(), gr.skip()
        
        # System status refresh
        refresh_system_btn.click(
            fn=lambda: panel.get_system_status(),
            outputs=[system_status]
        )
        
        # Periodic updates (every 5 seconds during training)
        train_panel.load(
            fn=auto_refresh,
            outputs=[training_output, progress_output, gpu_output],
            every=5
        )
    
    return train_panel