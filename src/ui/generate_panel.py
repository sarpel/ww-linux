"""
Generate Panel - TTS-based wake word audio generation interface
"""

import gradio as gr
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import tempfile
import os

# Import local modules
from ..generators.piper_tts_generator import PiperTTSGenerator
from ..generators.elevenlabs_generator import ElevenLabsGenerator
from ..utils.audio_converter import AudioConverter
from ..utils.file_manager import PanelCommunicator

logger = logging.getLogger(__name__)


class GeneratePanel:
    """Generate panel controller"""
    
    def __init__(self, communicator: PanelCommunicator):
        self.communicator = communicator
        self.piper_generator = PiperTTSGenerator()
        self.elevenlabs_generator = ElevenLabsGenerator()
        self.audio_converter = AudioConverter()
        self.output_dir = Path("data/generated")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_audio(
        self,
        wake_word: str,
        voice_type: str,
        tts_engine: str,
        elevenlabs_voice: str,
        num_samples: int,
        sample_rate: int,
        duration: float,
        progress=gr.Progress()
    ) -> Tuple[str, List[str]]:
        """
        Generate wake word audio samples
        
        Args:
            wake_word: Wake word text
            voice_type: Voice selection (for Piper)
            tts_engine: TTS engine selection (Piper or ElevenLabs)
            elevenlabs_voice: ElevenLabs voice selection
            num_samples: Number of samples to generate
            sample_rate: Audio sample rate
            duration: Duration per sample
            progress: Gradio progress tracker
            
        Returns:
            Tuple[str, List[str]]: Status message and list of generated files
        """
        try:
            if not wake_word.strip():
                return "<div class=\'error-panel\'>❌ Error: Please enter a wake word</div>", []
            
            progress(0, desc="Initializing generation...")
            
            # Select TTS engine and generate
            if tts_engine == "Piper TTS (GPU)":
                # Check if Piper TTS is available
                if not self.piper_generator.is_available():
                    return "<div class='error-panel'>❌ Piper TTS not available. Please install CUDA toolkit and Piper TTS models.</div>", []
                
                progress(0.5, desc="🚀 Generating batch with Piper TTS (GPU)...")
                successful_files, failed_files = self.piper_generator.generate_batch(
                    text=wake_word,
                    output_dir=self.output_dir,
                    batch_size=num_samples,
                    voice_variations=True
                )
                
            elif tts_engine == "ElevenLabs API":
                # Check if ElevenLabs is available
                if not self.elevenlabs_generator.is_available():
                    return "<div class='error-panel'>❌ ElevenLabs API not available. Please set ELEVENLABS_API_KEY environment variable.</div>", []
                
                progress(0.5, desc="🎭 Generating batch with ElevenLabs...")
                successful_files, failed_files = self.elevenlabs_generator.generate_batch(
                    text=wake_word,
                    output_dir=self.output_dir,
                    batch_size=num_samples,
                    voice_variations=True,
                    settings_variations=True
                )
                
            else:
                return "<div class='error-panel'>❌ Invalid TTS engine selected</div>", []
            
            generated_files = successful_files
            
            if failed_files:
                logger.warning(f"Failed to generate {len(failed_files)} samples: {failed_files}")
            
            if generated_files:
                # Trigger transfer to augment panel
                progress(1.0, desc="Transferring to Augment panel...")
                
                transfer_success = self.communicator.trigger_generate_to_augment()
                
                if transfer_success:
                    status = f"<div class='success-panel'>✅ Generated {len(generated_files)} samples and transferred to Augment panel</div>"
                else:
                    status = f"<div class='warning-panel'>⚠️ Generated {len(generated_files)} samples (transfer to Augment failed)</div>"
                
                return status, generated_files
            else:
                return "<div class='error-panel'>❌ Failed to generate any samples</div>", []
                
        except Exception as e:
            logger.error(f"Error in generate_audio: {e}")
            return f"<div class='error-panel'>❌ Error: {str(e)}</div>", []
    
    def get_generator_status(self) -> str:
        """Get status information about the TTS generators"""
        try:
            piper_status = self.piper_generator.get_status()
            elevenlabs_status = self.elevenlabs_generator.get_status()
            
            status_html = ""
            
            # Piper TTS Status
            if piper_status['available']:
                models_info = f"Models: {', '.join(piper_status['models'][:3])}"
                if len(piper_status['models']) > 3:
                    models_info += f" (+{len(piper_status['models'])-3} more)"
                
                status_html += f"""
                <div class='success-panel'>
                🚀 **Piper TTS (GPU-Only) Available**
                - GPU Support: {'✅' if piper_status['gpu_available'] else '❌'}
                - Models: {piper_status['num_models']} available
                - Default: {piper_status['default_model']}
                - {models_info}
                </div>
                """
            else:
                gpu_msg = "🚨 GPU Required" if not piper_status['gpu_available'] else "📦 Models Missing"
                status_html += f"""
                <div class='warning-panel'>
                ⚠️ **Piper TTS Not Available** - {gpu_msg}
                - GPU Available: {'✅' if piper_status['gpu_available'] else '❌ Install CUDA toolkit'}
                - API Available: {'✅' if piper_status['api_available'] else '❌ pip install piper-tts'}
                - Models: {piper_status['num_models']} found
                </div>
                """
            
            # ElevenLabs Status
            if elevenlabs_status['available']:
                voices_info = f"Voices: {len(elevenlabs_status['voices'])} available"
                status_html += f"""
                <div class='success-panel'>
                🎭 **ElevenLabs API Available**
                - API Key: ✅ Configured
                - {voices_info}
                - Default Voice: {elevenlabs_status['default_voice'] or 'Auto-selected'}
                </div>
                """
            else:
                status_html += """
                <div class='warning-panel'>
                ⚠️ **ElevenLabs API Not Available**
                - Set ELEVENLABS_API_KEY environment variable
                - Install: pip install elevenlabs
                </div>
                """
            
            return status_html
        except Exception as e:
            logger.error(f"Error getting generator status: {e}")
            return f"<div class='error-panel'>❌ Error checking status: {e}</div>"
    
    def get_available_models(self) -> List[str]:
        """Get list of available Piper TTS models"""
        try:
            return self.piper_generator.get_available_models()
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    def get_available_elevenlabs_voices(self) -> List[str]:
        """Get list of available ElevenLabs voices with names"""
        try:
            voice_names = self.elevenlabs_generator.get_voice_names()
            return [f"{name} ({voice_id[:8]}...)" for voice_id, name in voice_names.items()]
        except Exception as e:
            logger.error(f"Error getting ElevenLabs voices: {e}")
            return []
    
    def preview_audio(self, generated_files: List[str], selected_index: int) -> Optional[str]:
        """Preview a generated audio file"""
        try:
            if not generated_files or selected_index >= len(generated_files):
                return None
            
            file_path = generated_files[selected_index]
            if os.path.exists(file_path):
                return file_path
            
            return None
            
        except Exception as e:
            logger.error(f"Error in preview_audio: {e}")
            return None
    
    def get_generation_stats(self) -> str:
        """Get statistics about generated files"""
        try:
            stats = self.communicator.file_manager.get_folder_stats(self.output_dir)
            
            if stats['audio_files'] > 0:
                return f"""
                <div class='info-panel'>
                    <h4>📊 Generation Statistics</h4>
                    <div style='display: flex; gap: 15px; flex-wrap: wrap; margin-top: 10px;'>
                        <div class='metric-box'>
                            <div class='metric-value'>{stats['audio_files']}</div>
                            <div class='metric-label'>Audio Files</div>
                        </div>
                        <div class='metric-box'>
                            <div class='metric-value'>{stats['folder_size_mb']:.1f}</div>
                            <div class='metric-label'>Size (MB)</div>
                        </div>
                        <div class='metric-box'>
                            <div class='metric-value'>{stats['last_modified'] or 'Unknown'}</div>
                            <div class='metric-label'>Last Generated</div>
                        </div>
                    </div>
                </div>
                """
            else:
                return "<div class='info-panel'>📝 No audio files generated yet.</div>"
                
        except Exception as e:
            logger.error(f"Error getting generation stats: {e}")
            return "<div class='error-panel'>❌ Error getting statistics</div>"
    
    def clear_generated_files(self) -> str:
        """Clear all generated files"""
        try:
            removed, total = self.communicator.file_manager.clean_directory(
                self.output_dir,
                file_patterns=["*.wav", "*.pcm"]
            )
            
            if removed > 0:
                return f"<div class='success-panel'>✅ Cleared {removed} files</div>"
            else:
                return "<div class='info-panel'>📝 No files to clear</div>"
                
        except Exception as e:
            logger.error(f"Error clearing files: {e}")
            return f"<div class='error-panel'>❌ Error: {str(e)}</div>"


def create_generate_panel(communicator: PanelCommunicator):
    """Create the generate panel UI components"""
    
    panel = GeneratePanel(communicator)
    
    with gr.Column(elem_classes=["panel-container"]):
        
        # Input section
        with gr.Group():
            gr.Markdown("### 🎯 Wake Word Configuration")
            
            with gr.Row():
                wake_word_input = gr.Textbox(
                    label="Wake Word",
                    placeholder="Enter wake word (e.g., 'hey assistant', 'wake up')",
                    value="hey katya"
                )
                
                tts_engine = gr.Dropdown(
                    label="TTS Engine",
                    choices=["Piper TTS (GPU)", "ElevenLabs API"],
                    value="Piper TTS (GPU)"
                )
            
            with gr.Row():
                voice_type = gr.Dropdown(
                    label="Voice Type (Piper)",
                    choices=["female", "male", "neutral"],
                    value="female"
                )
                
                elevenlabs_voice = gr.Dropdown(
                    label="ElevenLabs Voice",
                    choices=panel.get_available_elevenlabs_voices(),
                    value=None,
                    visible=False
                )
            
            with gr.Row():
                num_samples = gr.Slider(
                    label="Number of Samples",
                    minimum=1,
                    maximum=500,
                    value=10,
                    step=1
                )
                
                sample_rate = gr.Dropdown(
                    label="Sample Rate (Hz)",
                    choices=[8000, 16000, 22050, 44100],
                    value=16000
                )
            
            duration = gr.Slider(
                label="Duration per Sample (seconds)",
                minimum=0.5,
                maximum=5.0,
                value=1.5,
                step=0.1
            )
        
        # Generation controls
        with gr.Group():
            gr.Markdown("### 🎵 Audio Generation")
            
            generate_btn = gr.Button(
                "🚀 Generate Wake Word Audio",
                variant="primary",
                size="lg"
            )
            
            generation_status = gr.HTML(
                value="<div class='info-panel'>🎙️ Ready to generate audio samples</div>"
            )
        
        # Preview section
        with gr.Group():
            gr.Markdown("### 👁️ Preview Generated Audio")
            
            with gr.Row():
                preview_dropdown = gr.Dropdown(
                    label="Select Sample to Preview",
                    choices=[],
                    interactive=True
                )
                
                preview_audio = gr.Audio(
                    label="Audio Preview",
                    interactive=False
                )
            
            with gr.Row():
                stats_display = gr.HTML(
                    value=panel.get_generation_stats()
                )
        
        # Management controls
        with gr.Group():
            gr.Markdown("### 🔧 File Management")
            
            with gr.Row():
                refresh_stats_btn = gr.Button("🔄 Refresh Stats", variant="secondary")
                clear_files_btn = gr.Button("🗑️ Clear Generated Files", variant="stop")
        
        # Hidden state for generated files
        generated_files_state = gr.State([])
        
        # Event handlers
        def on_generate(wake_word, voice_type, tts_engine, elevenlabs_voice, num_samples, sample_rate, duration, progress=gr.Progress()):
            status, files = panel.generate_audio(
                wake_word, voice_type, tts_engine, elevenlabs_voice or "",
                int(num_samples), int(sample_rate), duration, progress
            )
            
            # Update preview dropdown choices
            if files:
                choices = [f"Sample {i+1}: {Path(f).name}" for i, f in enumerate(files)]
            else:
                choices = []
            
            return status, files, gr.update(choices=choices, value=None), None
        
        def on_tts_engine_change(engine):
            if engine == "ElevenLabs API":
                return gr.update(visible=False), gr.update(visible=True, choices=panel.get_available_elevenlabs_voices())
            else:
                return gr.update(visible=True), gr.update(visible=False)
        
        def on_preview_select(files, selected):
            if not files or not selected:
                return None
            
            # Extract index from selection
            try:
                index = int(selected.split(":")[0].replace("Sample ", "")) - 1
                return panel.preview_audio(files, index)
            except:
                return None
        
        def on_refresh_stats():
            return panel.get_generation_stats()
        
        def on_clear_files():
            status = panel.clear_generated_files()
            stats = panel.get_generation_stats()
            return status, [], gr.update(choices=[], value=None), None, stats
        
        # Connect events
        tts_engine.change(
            fn=on_tts_engine_change,
            inputs=[tts_engine],
            outputs=[voice_type, elevenlabs_voice]
        )
        
        generate_btn.click(
            fn=on_generate,
            inputs=[wake_word_input, voice_type, tts_engine, elevenlabs_voice, num_samples, sample_rate, duration],
            outputs=[generation_status, generated_files_state, preview_dropdown, preview_audio]
        )
        
        preview_dropdown.change(
            fn=on_preview_select,
            inputs=[generated_files_state, preview_dropdown],
            outputs=[preview_audio]
        )
        
        refresh_stats_btn.click(
            fn=on_refresh_stats,
            outputs=[stats_display]
        )
        
        clear_files_btn.click(
            fn=on_clear_files,
            outputs=[generation_status, generated_files_state, preview_dropdown, preview_audio, stats_display]
        )
    
    return {
        'wake_word_input': wake_word_input,
        'voice_type': voice_type,
        'tts_engine': tts_engine,
        'elevenlabs_voice': elevenlabs_voice,
        'num_samples': num_samples,
        'sample_rate': sample_rate,
        'duration': duration,
        'generate_btn': generate_btn,
        'generation_status': generation_status,
        'preview_dropdown': preview_dropdown,
        'preview_audio': preview_audio,
        'stats_display': stats_display,
        'generated_files_state': generated_files_state
    }