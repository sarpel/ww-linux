"""
Wake Word Generator - Linux Ubuntu Edition
Optimized for Ubuntu 20.04+ with full GPU training support
"""

import os
import warnings
import sys
from pathlib import Path

# Linux-optimized environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
warnings.filterwarnings('ignore')

import gradio as gr
import logging
import socket
import psutil

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.ui.generate_panel import create_generate_panel
from src.ui.augment_panel import create_augment_panel
from src.ui.train_panel_linux import create_train_panel_linux
from src.utils.file_manager import FileManager, PanelCommunicator
from src.utils.unified_config import get_config

# Initialize unified configuration
config = get_config()

# Setup logging with Linux optimizations
logging.basicConfig(
    level=getattr(logging, config.get('logging', 'level', default='INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('wakeword-linux.log')  # Log to file on Linux
    ]
)
logger = logging.getLogger(__name__)

# Global state
file_manager = FileManager()
communicator = PanelCommunicator(file_manager)


def get_system_info():
    """Get Linux system information"""
    try:
        import torch
        import tensorflow as tf
        
        info = {
            'platform': 'Linux Ubuntu',
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / 1e9, 1),
            'pytorch_version': torch.__version__,
            'tensorflow_version': tf.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
                'cuda_version': torch.version.cuda,
            })
            
        return info
    except Exception as e:
        logger.warning(f"Could not get system info: {e}")
        return {'platform': 'Linux Ubuntu', 'error': str(e)}


def create_main_interface():
    """Create the main three-panel Gradio interface optimized for Linux"""
    
    # Get system info
    sys_info = get_system_info()
    
    with gr.Blocks(
        title="Wake Word Generator - Linux Ubuntu Edition",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray", neutral_hue="slate").set(
            body_background_fill_dark="#0f1419",
            background_fill_primary_dark="#1a1d29",
            background_fill_secondary_dark="#242733",
            border_color_primary_dark="#374151",
            block_background_fill_dark="#1a1d29",
        ),
        css="""
        /* Linux-optimized dark theme styling */
        .panel-container { 
            border: 2px solid #374151; 
            border-radius: 8px; 
            padding: 20px; 
            margin: 10px 0;
            background: var(--background-fill-secondary);
        }
        .panel-title {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 15px;
            color: var(--body-text-color);
            text-align: center;
        }
        .linux-info-panel {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid #10b981;
            border-radius: 8px;
            padding: 12px;
            margin: 10px 0;
            color: #34d399;
            font-family: monospace;
        }
        .status-box {
            background: var(--background-fill-primary);
            border: 1px solid var(--border-color-primary);
            border-radius: 8px;
            padding: 12px;
            margin: 10px 0;
            color: var(--body-text-color);
        }
        .flow-arrow {
            text-align: center;
            font-size: 2em;
            color: #10b981;
            margin: 20px 0;
            font-weight: bold;
        }
        """
    ) as interface:
        
        # Title and system info
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(
                    """
                    # 🐧 Wake Word Generator - Linux Ubuntu Edition
                    
                    **Optimized for Ubuntu 20.04+ with full GPU training support**
                    
                    Complete pipeline: Generate → Augment → **GPU Train** 🚀
                    """,
                    elem_classes=["panel-title"]
                )
            
            with gr.Column(scale=2):
                system_info_html = f"""
                <div class='linux-info-panel'>
                    <strong>🖥️ System Info</strong><br/>
                    Platform: {sys_info.get('platform', 'Unknown')}<br/>
                    Python: {sys_info.get('python_version', 'Unknown')}<br/>
                    CPUs: {sys_info.get('cpu_count', 'Unknown')}<br/>
                    Memory: {sys_info.get('memory_gb', 'Unknown')}GB<br/>
                """
                
                if sys_info.get('cuda_available'):
                    system_info_html += f"""
                    <strong>🔥 GPU Ready</strong><br/>
                    GPU: {sys_info.get('gpu_name', 'Unknown')}<br/>
                    VRAM: {sys_info.get('gpu_memory_gb', 'Unknown')}GB<br/>
                    CUDA: {sys_info.get('cuda_version', 'Unknown')}<br/>
                    """
                else:
                    system_info_html += "<strong>⚠️ No GPU detected</strong><br/>"
                
                system_info_html += "</div>"
                
                gr.HTML(system_info_html)
        
        # Status bar
        with gr.Row():
            status_display = gr.HTML(
                value="<div class='status-box'>🐧 Linux system ready. GPU training available!</div>",
                elem_classes=["status-box"]
            )
        
        # Refresh button
        refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary", size="sm")
        
        with gr.Tabs() as tabs:
            # Panel 1: Generate (unchanged but with Linux optimizations)
            with gr.TabItem("🎵 Generate", id="generate_tab"):
                gr.HTML("<div class='panel-title'>Step 1: Generate Wake Word Audio</div>")
                
                generate_components = create_generate_panel(communicator)
                
                gr.HTML("<div class='flow-arrow'>↓ Linux I/O Optimized → Augment Panel</div>")
            
            # Panel 2: Augment (unchanged)
            with gr.TabItem("🔄 Augment", id="augment_tab"):
                gr.HTML("<div class='panel-title'>Step 2: Upload & Augment Training Data</div>")
                
                augment_components = create_augment_panel(communicator)
                
                gr.HTML("<div class='flow-arrow'>↓ GPU-Ready Data → Training Panel</div>")
            
            # Panel 3: Train (Linux-optimized)
            with gr.TabItem("🚀 GPU Train", id="train_tab"):
                gr.HTML("<div class='panel-title'>Step 3: GPU-Accelerated Model Training</div>")
                
                train_components = create_train_panel_linux(communicator)
        
        # Global status refresh function
        def refresh_status():
            try:
                panel_status = file_manager.get_panel_status()
                
                status_parts = []
                
                # Generate panel status
                gen_stats = panel_status['generate']
                if gen_stats['audio_files'] > 0:
                    status_parts.append(f"✅ Generate: {gen_stats['audio_files']} files ({gen_stats['folder_size_mb']:.1f}MB)")
                else:
                    status_parts.append("⏸️ Generate: No files")
                
                # Augment panel status  
                aug_input = panel_status['augment_input']
                aug_pos = panel_status['augment_positive']
                aug_neg = panel_status['augment_negative']
                aug_proc = panel_status['augment_processed']
                
                augment_total = aug_input['audio_files'] + aug_pos['audio_files'] + aug_neg['audio_files']
                if augment_total > 0:
                    status_parts.append(f"✅ Augment: {augment_total} input files, {aug_proc['audio_files']} processed")
                else:
                    status_parts.append("⏸️ Augment: No files")
                
                # Train panel status
                train_stats = panel_status['training']
                models_stats = panel_status['models']
                if models_stats['file_count'] > 0:
                    status_parts.append(f"🚀 GPU Train: Model trained ({models_stats['file_count']} files)")
                elif train_stats['audio_files'] > 0:
                    status_parts.append(f"⚡ GPU Train: {train_stats['audio_files']} training files ready")
                else:
                    status_parts.append("⏸️ GPU Train: No training data")
                
                status_html = f"<div class='status-box'>🐧 {'  |  '.join(status_parts)}</div>"
                return status_html
                
            except Exception as e:
                logger.error(f"Error refreshing status: {e}")
                return "<div class='status-box'>❌ Error refreshing status</div>"
        
        # Connect refresh button
        refresh_btn.click(
            fn=refresh_status,
            outputs=[status_display]
        )
        
        # Setup panel callbacks
        def setup_panel_callbacks():
            """Setup callbacks for inter-panel communication"""
            
            def on_files_transferred(data):
                logger.info(f"Linux I/O - Files transferred: {data}")
            
            def on_training_data_ready(data):
                logger.info(f"GPU Training - Data ready: {data}")
            
            communicator.register_callback('files_transferred', on_files_transferred)
            communicator.register_callback('training_data_ready', on_training_data_ready)
        
        # Setup callbacks and auto-refresh
        interface.load(setup_panel_callbacks)
        interface.load(
            fn=refresh_status,
            outputs=[status_display]
        )
    
    return interface


def find_available_port(start_port=7860, end_port=7870):
    """Find an available port optimized for Linux"""
    for port in range(start_port, end_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Linux optimization
                s.bind(('0.0.0.0', port))
                logger.info(f"🔌 Found available port: {port}")
                return port
        except OSError:
            continue
    
    logger.warning(f"No ports available in range {start_port}-{end_port}, using random port")
    return 0


def main():
    """Main entry point for Linux Ubuntu edition"""
    try:
        logger.info("🐧 Starting Wake Word Generator - Linux Ubuntu Edition...")
        
        # Get system info
        sys_info = get_system_info()
        logger.info(f"🖥️ System: {sys_info.get('platform')} | Python {sys_info.get('python_version')}")
        logger.info(f"💾 Memory: {sys_info.get('memory_gb')}GB | CPUs: {sys_info.get('cpu_count')}")
        
        if sys_info.get('cuda_available'):
            logger.info(f"🚀 GPU: {sys_info.get('gpu_name')} | VRAM: {sys_info.get('gpu_memory_gb')}GB")
            logger.info(f"🔥 CUDA: {sys_info.get('cuda_version')} | GPUs: {sys_info.get('gpu_count')}")
        else:
            logger.warning("⚠️ No CUDA GPU detected - training will not be available")
        
        # Configuration
        host = config.get('application', 'host', default="0.0.0.0")
        config_port = config.get('application', 'port', default=7860)
        debug = config.get('application', 'debug', default=False)
        
        # Find available port
        port = find_available_port(config_port, config_port + 10)
        
        # Ensure data directories exist
        file_manager._ensure_base_structure()
        
        # Create and launch interface
        interface = create_main_interface()
        
        # Enable queuing for progress tracking
        interface.queue()
        
        # Launch with Linux optimizations
        logger.info(f"🚀 Launching on {host}:{port}...")
        
        launch_kwargs = {
            'server_name': host,
            'share': False,
            'quiet': not debug,
            'show_error': True,
            'max_threads': psutil.cpu_count(),  # Linux optimization
        }
        
        if port > 0:
            launch_kwargs['server_port'] = port
        
        try:
            interface.launch(**launch_kwargs)
        except OSError as port_error:
            logger.warning(f"Port {port} unavailable: {port_error}")
            launch_kwargs.pop('server_port', None)
            interface.launch(**launch_kwargs)
        
    except Exception as e:
        logger.error(f"💥 Error launching application: {e}")
        raise


if __name__ == "__main__":
    main()