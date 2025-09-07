"""
Audio conversion utilities for PCM to WAV conversion and format handling
"""

import os
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AudioConverter:
    """Audio format conversion utilities with PCM to WAV specialization"""
    
    def __init__(self):
        self.supported_formats = ['.pcm', '.wav', '.mp3', '.flac', '.ogg']
        
    def convert_pcm_to_wav(
        self, 
        pcm_file: Union[str, Path], 
        output_file: Union[str, Path],
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = 'int16'
    ) -> bool:
        """
        Convert PCM audio file to WAV format
        
        Args:
            pcm_file: Path to input PCM file
            output_file: Path to output WAV file
            sample_rate: Sample rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1)
            dtype: Data type for PCM data (default: 'int16')
            
        Returns:
            bool: True if conversion successful, False otherwise
        """
        try:
            pcm_file = Path(pcm_file)
            output_file = Path(output_file)
            
            if not pcm_file.exists():
                logger.error(f"PCM file not found: {pcm_file}")
                return False
                
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Read raw PCM data
            with open(pcm_file, 'rb') as f:
                raw_data = f.read()
            
            # Convert bytes to numpy array based on dtype
            if dtype == 'int16':
                audio_data = np.frombuffer(raw_data, dtype=np.int16)
            elif dtype == 'int32':
                audio_data = np.frombuffer(raw_data, dtype=np.int32)
            elif dtype == 'float32':
                audio_data = np.frombuffer(raw_data, dtype=np.float32)
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")
            
            # Reshape for multi-channel audio
            if channels > 1:
                if len(audio_data) % channels != 0:
                    # Trim to make divisible by channels
                    trim_length = len(audio_data) - (len(audio_data) % channels)
                    audio_data = audio_data[:trim_length]
                audio_data = audio_data.reshape(-1, channels)
            
            # Normalize if integer type
            if dtype in ['int16', 'int32']:
                audio_data = audio_data.astype(np.float32)
                if dtype == 'int16':
                    audio_data /= 32767.0
                elif dtype == 'int32':
                    audio_data /= 2147483647.0
            
            # Write WAV file
            sf.write(output_file, audio_data, sample_rate)
            
            logger.info(f"Successfully converted {pcm_file} to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error converting PCM to WAV: {e}")
            return False
    
    def convert_to_standard_format(
        self,
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        target_sample_rate: int = 16000,
        target_channels: int = 1
    ) -> bool:
        """
        Convert any supported audio format to standardized WAV
        
        Args:
            input_file: Path to input audio file
            output_file: Path to output WAV file
            target_sample_rate: Target sample rate in Hz
            target_channels: Target number of channels
            
        Returns:
            bool: True if conversion successful, False otherwise
        """
        try:
            input_file = Path(input_file)
            output_file = Path(output_file)
            
            if not input_file.exists():
                logger.error(f"Input file not found: {input_file}")
                return False
            
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Load audio file
            audio_data, sample_rate = librosa.load(
                input_file, 
                sr=target_sample_rate,
                mono=(target_channels == 1)
            )
            
            # Convert to target channels if needed
            if target_channels == 2 and len(audio_data.shape) == 1:
                # Convert mono to stereo
                audio_data = np.stack([audio_data, audio_data], axis=1)
            elif target_channels == 1 and len(audio_data.shape) == 2:
                # Convert stereo to mono (already handled by librosa.load with mono=True)
                pass
            
            # Write standardized WAV file
            sf.write(output_file, audio_data, target_sample_rate)
            
            logger.info(f"Successfully converted {input_file} to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error converting audio format: {e}")
            return False
    
    def batch_convert_pcm_to_wav(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = 'int16'
    ) -> Tuple[int, int]:
        """
        Batch convert all PCM files in directory to WAV format
        
        Args:
            input_dir: Directory containing PCM files
            output_dir: Directory for output WAV files
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
            dtype: Data type for PCM data
            
        Returns:
            Tuple[int, int]: (successful_conversions, total_files)
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return 0, 0
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pcm_files = list(input_dir.glob('*.pcm'))
        successful = 0
        
        for pcm_file in pcm_files:
            output_file = output_dir / f"{pcm_file.stem}.wav"
            if self.convert_pcm_to_wav(pcm_file, output_file, sample_rate, channels, dtype):
                successful += 1
        
        logger.info(f"Batch conversion complete: {successful}/{len(pcm_files)} files converted")
        return successful, len(pcm_files)
    
    def get_audio_info(self, file_path: Union[str, Path]) -> Optional[dict]:
        """
        Get audio file information
        
        Args:
            file_path: Path to audio file
            
        Returns:
            dict: Audio file information or None if error
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return None
            
            # Use soundfile for WAV files, librosa for others
            if file_path.suffix.lower() == '.wav':
                info = sf.info(file_path)
                return {
                    'duration': info.frames / info.samplerate,
                    'sample_rate': info.samplerate,
                    'channels': info.channels,
                    'frames': info.frames,
                    'format': info.format,
                    'subtype': info.subtype
                }
            else:
                # Use librosa for other formats
                duration = librosa.get_duration(filename=file_path)
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                
                return {
                    'duration': duration,
                    'sample_rate': sample_rate,
                    'channels': 1 if len(audio_data.shape) == 1 else audio_data.shape[1],
                    'frames': len(audio_data),
                    'format': file_path.suffix[1:].upper(),
                    'subtype': None
                }
                
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            return None
    
    def validate_audio_file(self, file_path: Union[str, Path]) -> bool:
        """
        Validate if file is a valid audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            bool: True if valid audio file, False otherwise
        """
        try:
            info = self.get_audio_info(file_path)
            return info is not None and info['duration'] > 0
        except:
            return False


def create_silence(
    duration: float,
    sample_rate: int = 16000,
    channels: int = 1
) -> np.ndarray:
    """
    Create silence audio data
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        channels: Number of channels
        
    Returns:
        np.ndarray: Silent audio data
    """
    frames = int(duration * sample_rate)
    if channels == 1:
        return np.zeros(frames, dtype=np.float32)
    else:
        return np.zeros((frames, channels), dtype=np.float32)


def normalize_audio(audio_data: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target decibel level
    
    Args:
        audio_data: Input audio data
        target_db: Target decibel level
        
    Returns:
        np.ndarray: Normalized audio data
    """
    # Calculate RMS
    rms = np.sqrt(np.mean(audio_data ** 2))
    
    if rms == 0:
        return audio_data
    
    # Convert target dB to linear scale
    target_rms = 10 ** (target_db / 20.0)
    
    # Apply normalization
    return audio_data * (target_rms / rms)


if __name__ == "__main__":
    # Example usage
    converter = AudioConverter()
    
    # Test PCM to WAV conversion
    # converter.convert_pcm_to_wav(
    #     "input.pcm", 
    #     "output.wav", 
    #     sample_rate=16000, 
    #     channels=1, 
    #     dtype='int16'
    # )
    
    print("Audio converter utilities ready!")