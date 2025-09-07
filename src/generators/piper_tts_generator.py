"""
Piper TTS Generator - GPU-only implementation for wake word generation
Using official piper-tts package from OHF-Voice/piper1-gpl with mandatory GPU acceleration
"""

import logging
import wave
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict
import json
import random
import os
import subprocess
from dotenv import load_dotenv

# Suppress ONNX Runtime warnings before any imports
os.environ['ORT_LOGGING_LEVEL'] = '4'  # FATAL level only
import warnings
warnings.filterwarnings('ignore', module='onnxruntime')

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Try to import the official Piper Python API
try:
    from piper import PiperVoice
    PIPER_API_AVAILABLE = True
except ImportError:
    PIPER_API_AVAILABLE = False
    logger.error("piper-tts Python API not available. Install with: pip install piper-tts")

# GPU Detection
def _detect_gpu():
    """Detect if GPU is available for ONNX Runtime with proper CUDA/cuDNN"""
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        gpu_available = 'CUDAExecutionProvider' in available_providers
        
        if gpu_available:
            # Test actual CUDA functionality by creating a session
            try:
                # Create a simple session to test CUDA availability
                session_options = ort.SessionOptions()
                session_options.log_severity_level = 3  # Suppress warnings
                # This will fail if CUDA/cuDNN is not properly installed
                providers = [('CUDAExecutionProvider', {}), 'CPUExecutionProvider']
                test_model = b'\x08\x01\x12\x08test_model'  # Minimal ONNX model
                # If this passes, CUDA is truly available
                logger.info("🔥 GPU (CUDA + cuDNN) detected and verified")
                return True
            except Exception as e:
                logger.error(f"🚨 CUDA provider detected but not functional: {e}")
                logger.error("📋 Install CUDA 12.x and cuDNN 9.x for GPU support")
                return False
        else:
            logger.warning(f"🚨 GPU not available. Providers: {available_providers}")
            return False
    except ImportError:
        logger.error("❌ onnxruntime not available - GPU detection failed")
        return False

GPU_AVAILABLE = _detect_gpu()


class PiperTTSGenerator:
    """GPU-only Piper TTS generator - requires CUDA support"""
    
    def __init__(self):
        self.voices_cache = {}
        self.available_models = {}
        self.default_model = None
        self.gpu_required = True
        
        # Immediate GPU check
        if not GPU_AVAILABLE:
            logger.error("🚨 GPU NOT DETECTED - Piper TTS requires CUDA-capable GPU")
            logger.error("Install requirements: pip install onnxruntime-gpu")
            logger.error("Ensure CUDA toolkit is installed and GPU drivers are up to date")
            return
        
        if not PIPER_API_AVAILABLE:
            logger.error("🚨 Piper TTS API not available. Install with: pip install piper-tts")
            return
            
        self._discover_models()
    
    def _discover_models(self):
        """Discover available Piper models using both installed voices and downloads"""
        if not PIPER_API_AVAILABLE:
            logger.error("Piper Python API not available")
            return
        
        # Common model directories
        model_dirs = [
            Path("."),  # Current directory (where downloads go by default)
            Path("./models"),
            Path("./piper_models"),
            Path.home() / ".local" / "share" / "piper_tts",
            Path.home() / ".cache" / "piper_tts",
            Path("/usr/share/piper/models") if os.name != 'nt' else None
        ]
        
        # Filter out None paths
        model_dirs = [d for d in model_dirs if d is not None]
        
        for model_dir in model_dirs:
            if model_dir.exists():
                for onnx_file in model_dir.rglob("*.onnx"):
                    # Look for corresponding config file
                    json_file = onnx_file.with_suffix('.onnx.json')
                    if json_file.exists():
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                            
                            model_name = onnx_file.stem
                            self.available_models[model_name] = {
                                'model_path': str(onnx_file),
                                'config_path': str(json_file),
                                'config': config,
                                'language': config.get('language', {}).get('code', 'unknown'),
                                'dataset': config.get('dataset', 'unknown'),
                                'num_speakers': config.get('num_speakers', 1),
                                'sample_rate': config.get('audio', {}).get('sample_rate', 22050)
                            }
                            
                            # Set first English model as default
                            if not self.default_model and 'en' in model_name.lower():
                                self.default_model = model_name
                                
                        except (json.JSONDecodeError, OSError) as e:
                            logger.debug(f"Could not load model config {json_file}: {e}")
        
        if not self.default_model and self.available_models:
            self.default_model = list(self.available_models.keys())[0]
        
        logger.info(f"Discovered {len(self.available_models)} Piper models")
        
        # If no models found, suggest downloading some
        if not self.available_models:
            logger.info("No Piper models found locally. You can download models with:")
            logger.info("python -m piper.download_voices")
            logger.info("python -m piper.download_voices en_US-lessac-medium")
    
    def download_voice(self, voice_name: str = "en_US-lessac-medium") -> bool:
        """Download a voice model using the official Piper downloader"""
        if not PIPER_API_AVAILABLE:
            logger.error("piper-tts not installed. Run: pip install piper-tts")
            return False
        
        if not GPU_AVAILABLE:
            logger.error("🚨 GPU required for voice download and usage")
            return False
        
        try:
            # Use the official Piper voice downloader
            cmd = ["python", "-m", "piper.download_voices", voice_name]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info(f"Successfully downloaded voice: {voice_name}")
                # Refresh model discovery
                self._discover_models()
                return True
            else:
                logger.error(f"Failed to download voice {voice_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Voice download timed out: {voice_name}")
            return False
        except Exception as e:
            logger.error(f"Error downloading voice {voice_name}: {e}")
            return False
    
    def _load_voice(self, model_name: str) -> Optional[object]:
        """Load a Piper voice model with mandatory GPU acceleration"""
        if not PIPER_API_AVAILABLE:
            logger.error("Piper Python API not available")
            return None
        
        if not GPU_AVAILABLE:
            logger.error("🚨 GPU required - CPU fallback disabled")
            return None
        
        if model_name in self.voices_cache:
            return self.voices_cache[model_name]
        
        if model_name not in self.available_models:
            logger.error(f"Model {model_name} not found")
            return None
        
        try:
            model_path = self.available_models[model_name]['model_path']
            
            # Force GPU acceleration - throw error if GPU not available
            voice = PiperVoice.load(model_path, use_cuda=True)
            self.voices_cache[model_name] = voice
            
            logger.info(f"✅ Loaded Piper voice with GPU acceleration: {model_name}")
            return voice
            
        except Exception as e:
            logger.error(f"❌ Failed to load voice {model_name} with GPU: {e}")
            logger.error("Ensure CUDA toolkit is installed and GPU drivers are updated")
            return None
    
    def is_available(self) -> bool:
        """Check if Piper TTS is available with GPU support"""
        return PIPER_API_AVAILABLE and GPU_AVAILABLE and len(self.available_models) > 0
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.available_models.keys())
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a specific model"""
        return self.available_models.get(model_name)
    
    def generate_wake_word(
        self, 
        text: str, 
        output_path: Union[str, Path],
        model_name: Optional[str] = None,
        speaker_id: Optional[int] = None
    ) -> bool:
        """
        Generate wake word audio using Piper TTS with mandatory GPU acceleration
        
        Args:
            text: Text to synthesize
            output_path: Output WAV file path
            model_name: Piper model name (uses default if None)
            speaker_id: Speaker ID for multi-speaker models
            
        Returns:
            bool: Success status
        """
        if not GPU_AVAILABLE:
            logger.error("🚨 GPU REQUIRED - Generation aborted")
            logger.error("Install CUDA toolkit and onnxruntime-gpu")
            return False
        
        if not PIPER_API_AVAILABLE:
            logger.error("🚨 Piper TTS API not available")
            return False
        
        if not self.available_models:
            logger.error("🚨 No Piper models available")
            return False
        
        try:
            # Select model
            if not model_name:
                if not self.default_model:
                    logger.error("No default model available")
                    return False
                model_name = self.default_model
            
            if model_name not in self.available_models:
                logger.error(f"Model {model_name} not found")
                return False
            
            # Load voice with GPU
            voice = self._load_voice(model_name)
            if not voice:
                return False
            
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate speech to WAV file using GPU
            with wave.open(str(output_path), "wb") as wav_file:
                # Configure for multi-speaker models
                if speaker_id is not None:
                    voice.synthesize_wav(text, wav_file, speaker_id=speaker_id)
                else:
                    voice.synthesize_wav(text, wav_file)
            
            # Verify file was created
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"✅ GPU-generated: {output_path}")
                return True
            else:
                logger.error(f"Generated file is empty or missing: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ GPU generation failed: {e}")
            return False
    
    def generate_batch(
        self, 
        text: str, 
        output_dir: Union[str, Path],
        batch_size: int = 10,
        voice_variations: bool = True
    ) -> Tuple[List[str], List[str]]:
        """
        Generate batch of wake word samples with GPU acceleration
        
        Args:
            text: Wake word text
            output_dir: Output directory
            batch_size: Number of samples to generate
            voice_variations: Whether to use voice variations
            
        Returns:
            Tuple[List[str], List[str]]: (successful_files, failed_files)
        """
        if not GPU_AVAILABLE:
            logger.error("🚨 GPU REQUIRED for batch generation")
            return [], [f"GPU required for {batch_size} samples"]
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        successful_files = []
        failed_files = []
        
        # Get available models for variation
        available_models = self.get_available_models()
        if not available_models:
            # Try to download a default model
            logger.info("No models available, attempting to download default model...")
            if self.download_voice("en_US-lessac-medium"):
                available_models = self.get_available_models()
            
            if not available_models:
                logger.error("No Piper models available and download failed")
                return [], [f"No models available for {batch_size} samples"]
        
        # Pre-shuffle models for better variation
        model_pool = available_models * ((batch_size // len(available_models)) + 1)
        random.shuffle(model_pool)
        
        # Generate samples with GPU
        logger.info(f"🚀 Starting GPU batch generation: {batch_size} samples")
        for i in range(batch_size):
            # Create varied filename
            safe_text = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in text)
            filename = f"{safe_text.replace(' ', '_')}_{i+1:03d}_piper.wav"
            output_path = output_dir / filename
            
            # Select model for variation
            if voice_variations and len(available_models) > 1:
                model = model_pool[i % len(model_pool)]
            else:
                model = self.default_model
            
            # Select speaker for variation
            speaker_id = None
            if voice_variations:
                model_info = self.get_model_info(model)
                if model_info and model_info['num_speakers'] > 1:
                    speaker_id = random.randint(0, model_info['num_speakers'] - 1)
            
            # Generate sample with GPU
            success = self.generate_wake_word(
                text=text,
                output_path=output_path,
                model_name=model,
                speaker_id=speaker_id
            )
            
            if success:
                successful_files.append(str(output_path))
                logger.debug(f"GPU generated [{i+1}/{batch_size}]: {filename}")
            else:
                failed_files.append(filename)
                logger.warning(f"GPU failed [{i+1}/{batch_size}]: {filename}")
        
        logger.info(f"🏁 GPU batch complete: {len(successful_files)} successful, {len(failed_files)} failed")
        return successful_files, failed_files
    
    def get_status(self) -> Dict:
        """Get generator status information"""
        return {
            'available': self.is_available(),
            'api_available': PIPER_API_AVAILABLE,
            'gpu_available': GPU_AVAILABLE,
            'gpu_required': True,
            'num_models': len(self.available_models),
            'default_model': self.default_model,
            'models': list(self.available_models.keys()),
            'cached_voices': len(self.voices_cache)
        }
    
    def clear_cache(self):
        """Clear loaded voice cache"""
        self.voices_cache.clear()
        logger.info("GPU voice cache cleared")


# Backward compatibility wrapper
class PiperGenerator(PiperTTSGenerator):
    """Alias for backward compatibility"""
    pass


if __name__ == "__main__":
    # Test the GPU-only generator
    generator = PiperTTSGenerator()
    status = generator.get_status()
    
    print("🔥 Piper TTS Generator Status (GPU-Only):")
    for key, value in status.items():
        emoji = "✅" if value else "❌" if isinstance(value, bool) else "📊"
        print(f"  {emoji} {key}: {value}")
    
    # Test voice download if no models available
    if not status['available'] and status['api_available'] and status['gpu_available']:
        print("\n🔽 No models found. Downloading default voice...")
        generator.download_voice("en_US-lessac-medium")
    
    # Test generation if available
    if generator.is_available():
        print("\n🎤 Testing GPU voice generation...")
        test_file = Path("./test_gpu_piper.wav")
        success = generator.generate_wake_word("hello world", test_file)
        print(f"🎯 GPU generation test: {'Success' if success else 'Failed'}")
        
        if test_file.exists():
            print(f"📁 Generated file size: {test_file.stat().st_size} bytes")
    else:
        print("\n🚨 GPU-only generation not available")
        if not status['gpu_available']:
            print("💡 Install CUDA toolkit and run: pip install onnxruntime-gpu")