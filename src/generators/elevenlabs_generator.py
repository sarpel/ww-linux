"""
ElevenLabs Generator - Alternative TTS using ElevenLabs API
High-quality voice cloning and synthesis for wake word generation
"""

import logging
import requests
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict
import json
import random
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Try to import elevenlabs client
try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import VoiceSettings, Voice
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    logger.warning("ElevenLabs API not available. Install with: pip install elevenlabs")


class ElevenLabsGenerator:
    """ElevenLabs TTS generator for high-quality voice synthesis"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ELEVENLABS_API_KEY') or os.getenv('WAKEWORD_API_KEY')
        self.client = None
        self.available_voices = {}
        self.default_voice = None
        
        if not self.api_key:
            logger.error("🚨 ElevenLabs API key not found. Set ELEVENLABS_API_KEY or WAKEWORD_API_KEY")
            return
            
        if not ELEVENLABS_AVAILABLE:
            logger.error("🚨 ElevenLabs client not available. Install with: pip install elevenlabs")
            return
        
        # Initialize client
        try:
            self.client = ElevenLabs(api_key=self.api_key)
            # Discover available voices
            self._discover_voices()
        except Exception as e:
            logger.error(f"Failed to initialize ElevenLabs client: {e}")
            self.client = None
    
    def _discover_voices(self):
        """Discover available voices from ElevenLabs API"""
        if not self.client:
            return
        
        try:
            # Use ElevenLabs client to get voices
            voices_response = self.client.voices.get_all()
            
            for voice in voices_response.voices:
                voice_id = voice.voice_id
                voice_name = voice.name
                
                if voice_id and voice_name:
                    self.available_voices[voice_id] = {
                        'name': voice_name,
                        'voice_id': voice_id,
                        'category': getattr(voice, 'category', 'premade'),
                        'labels': getattr(voice, 'labels', {}),
                        'settings': getattr(voice, 'settings', {})
                    }
                    
                    # Set first voice as default
                    if not self.default_voice:
                        self.default_voice = voice_id
            
            logger.info(f"Discovered {len(self.available_voices)} ElevenLabs voices")
                
        except Exception as e:
            logger.error(f"Error discovering ElevenLabs voices: {e}")
    
    def is_available(self) -> bool:
        """Check if ElevenLabs is available"""
        return bool(self.client and ELEVENLABS_AVAILABLE and self.available_voices)
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voice IDs"""
        return list(self.available_voices.keys())
    
    def get_voice_info(self, voice_id: str) -> Optional[Dict]:
        """Get information about a specific voice"""
        return self.available_voices.get(voice_id)
    
    def get_voice_names(self) -> Dict[str, str]:
        """Get mapping of voice_id -> voice_name"""
        return {vid: info['name'] for vid, info in self.available_voices.items()}
    
    def generate_wake_word(
        self,
        text: str,
        output_path: Union[str, Path],
        voice_id: Optional[str] = None,
        stability: float = 0.75,
        similarity_boost: float = 0.85,
        style: float = 0.0,
        use_speaker_boost: bool = True
    ) -> bool:
        """
        Generate wake word audio using ElevenLabs API
        
        Args:
            text: Text to synthesize
            output_path: Output audio file path
            voice_id: ElevenLabs voice ID (uses default if None)
            stability: Voice stability (0.0-1.0)
            similarity_boost: Voice similarity boost (0.0-1.0)
            style: Style setting (0.0-1.0)
            use_speaker_boost: Enable speaker boost
            
        Returns:
            bool: Success status
        """
        if not self.is_available():
            logger.error("🚨 ElevenLabs not available - check API key and installation")
            return False
        
        try:
            # Select voice
            if not voice_id:
                if not self.default_voice:
                    logger.error("No default voice available")
                    return False
                voice_id = self.default_voice
            
            if voice_id not in self.available_voices:
                logger.error(f"Voice {voice_id} not found")
                return False
            
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Configure voice settings
            voice_settings = VoiceSettings(
                stability=stability,
                similarity_boost=similarity_boost,
                style=style,
                use_speaker_boost=use_speaker_boost
            )
            
            # Generate audio using ElevenLabs
            logger.info(f"🎙️ Generating with ElevenLabs voice: {self.available_voices[voice_id]['name']}")
            
            audio_generator = self.client.text_to_speech.generate(
                text=text,
                voice=voice_id,
                voice_settings=voice_settings,
                model="eleven_multilingual_v2"
            )
            
            # Save audio to file
            with open(output_path, 'wb') as f:
                for audio_chunk in audio_generator:
                    f.write(audio_chunk)
            
            # Verify file was created
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"✅ ElevenLabs generated: {output_path}")
                return True
            else:
                logger.error(f"Generated file is empty or missing: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ ElevenLabs generation failed: {e}")
            if "quota" in str(e).lower():
                logger.error("💸 ElevenLabs API quota exceeded")
            elif "unauthorized" in str(e).lower():
                logger.error("🔑 ElevenLabs API key invalid or expired")
            return False
    
    def generate_batch(
        self,
        text: str,
        output_dir: Union[str, Path],
        batch_size: int = 10,
        voice_variations: bool = True,
        settings_variations: bool = True
    ) -> Tuple[List[str], List[str]]:
        """
        Generate batch of wake word samples with ElevenLabs
        
        Args:
            text: Wake word text
            output_dir: Output directory
            batch_size: Number of samples to generate
            voice_variations: Whether to use different voices
            settings_variations: Whether to vary voice settings
            
        Returns:
            Tuple[List[str], List[str]]: (successful_files, failed_files)
        """
        if not self.is_available():
            logger.error("🚨 ElevenLabs not available for batch generation")
            return [], [f"ElevenLabs unavailable for {batch_size} samples"]
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        successful_files = []
        failed_files = []
        
        # Get available voices for variation
        available_voices = self.get_available_voices()
        if not available_voices:
            logger.error("No ElevenLabs voices available")
            return [], [f"No voices available for {batch_size} samples"]
        
        # Pre-shuffle voices for better variation
        if voice_variations:
            voice_pool = available_voices * ((batch_size // len(available_voices)) + 1)
            random.shuffle(voice_pool)
        else:
            voice_pool = [self.default_voice] * batch_size
        
        # Generate samples
        logger.info(f"🎭 Starting ElevenLabs batch generation: {batch_size} samples")
        for i in range(batch_size):
            # Create varied filename
            safe_text = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in text)
            filename = f"{safe_text.replace(' ', '_')}_{i+1:03d}_elevenlabs.wav"
            output_path = output_dir / filename
            
            # Select voice for variation
            voice_id = voice_pool[i % len(voice_pool)] if voice_variations else self.default_voice
            
            # Generate random settings if variations enabled
            if settings_variations:
                stability = random.uniform(0.55, 0.85)
                similarity_boost = random.uniform(0.75, 0.95)
                style = random.uniform(0.0, 0.25)
                use_speaker_boost = random.random() < 0.9
            else:
                stability = 0.75
                similarity_boost = 0.85
                style = 0.0
                use_speaker_boost = True
            
            # Generate sample
            success = self.generate_wake_word(
                text=text,
                output_path=output_path,
                voice_id=voice_id,
                stability=stability,
                similarity_boost=similarity_boost,
                style=style,
                use_speaker_boost=use_speaker_boost
            )
            
            if success:
                successful_files.append(str(output_path))
                logger.debug(f"ElevenLabs generated [{i+1}/{batch_size}]: {filename}")
            else:
                failed_files.append(filename)
                logger.warning(f"ElevenLabs failed [{i+1}/{batch_size}]: {filename}")
        
        logger.info(f"🎬 ElevenLabs batch complete: {len(successful_files)} successful, {len(failed_files)} failed")
        return successful_files, failed_files
    
    def get_status(self) -> Dict:
        """Get generator status information"""
        return {
            'available': self.is_available(),
            'api_available': ELEVENLABS_AVAILABLE,
            'api_key_set': bool(self.api_key),
            'client_initialized': bool(self.client),
            'num_voices': len(self.available_voices),
            'default_voice': self.default_voice,
            'voices': list(self.available_voices.keys())
        }


if __name__ == "__main__":
    # Test the ElevenLabs generator
    generator = ElevenLabsGenerator()
    status = generator.get_status()
    
    print("🎭 ElevenLabs Generator Status:")
    for key, value in status.items():
        emoji = "✅" if value else "❌" if isinstance(value, bool) else "📊"
        print(f"  {emoji} {key}: {value}")
    
    # Test generation if available
    if generator.is_available():
        print("\n🎤 Testing ElevenLabs voice generation...")
        test_file = Path("./test_elevenlabs.wav")
        success = generator.generate_wake_word("hello world", test_file)
        print(f"🎯 Generation test: {'Success' if success else 'Failed'}")
        
        if test_file.exists():
            print(f"📁 Generated file size: {test_file.stat().st_size} bytes")
    else:
        print("\n🚨 ElevenLabs generation not available")
        if not status['api_key_set']:
            print("💡 Set ELEVENLABS_API_KEY environment variable")
        if not status['api_available']:
            print("💡 Install ElevenLabs: pip install elevenlabs")