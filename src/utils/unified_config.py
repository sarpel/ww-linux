"""
Unified Configuration Manager
Handles all configuration loading with proper precedence: .env > config.json > defaults
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class UnifiedConfig:
    """Unified configuration manager with environment variable override support"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self._config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration with precedence: .env > config.json > defaults"""
        try:
            # Load base config from JSON
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
            else:
                logger.warning(f"Config file {self.config_file} not found, using defaults")
                self._config = self._get_default_config()
            
            # Override with environment variables
            self._apply_env_overrides()
            
            # Ensure required directories exist
            self._ensure_directories()
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._config = self._get_default_config()
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        env_mappings = {
            # API Configuration
            'WAKEWORD_API_KEY': ('elevenlabs', 'api_key'),
            
            # Directory Configuration  
            'WAKEWORD_OUTPUT_DIRECTORY': ('directories', 'generated'),
            'WAKEWORD_NOISE_DIRECTORY': ('directories', 'background_noise'),
            'WAKEWORD_CACHE_DIRECTORY': ('directories', 'cache'),
            
            # Performance Configuration
            'WAKEWORD_CUDA_BATCH_SIZE': ('performance', 'cuda', 'batch_size'),
            'WAKEWORD_NUM_WORKERS': ('performance', 'cuda', 'num_workers'),
            'WAKEWORD_MAX_FILE_SIZE_MB': ('performance', 'limits', 'max_file_size_mb'),
            'WAKEWORD_MAX_BATCH_SIZE': ('performance', 'limits', 'max_batch_size'),
            
            # Logging Configuration
            'WAKEWORD_LOG_LEVEL': ('logging', 'level'),
        }
        
        for env_key, config_path in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value:
                self._set_nested_value(config_path, env_value)
    
    def _set_nested_value(self, path: tuple, value: str):
        """Set a nested configuration value"""
        try:
            current = self._config
            for key in path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Type conversion
            final_key = path[-1]
            if isinstance(current.get(final_key), int):
                current[final_key] = int(value)
            elif isinstance(current.get(final_key), float):
                current[final_key] = float(value)
            elif isinstance(current.get(final_key), bool):
                current[final_key] = value.lower() in ('true', '1', 'yes', 'on')
            else:
                current[final_key] = value
                
        except Exception as e:
            logger.warning(f"Could not set config value {path}: {e}")
    
    def _ensure_directories(self):
        """Ensure all configured directories exist"""
        try:
            directories = self.get('directories')
            if directories and isinstance(directories, dict):
                for dir_key, dir_path in directories.items():
                    if dir_path:
                        Path(dir_path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Error creating directories: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "application": {
                "name": "Wake Word Generator",
                "version": "1.0.0",
                "host": "127.0.0.1",
                "port": 7860
            },
            "directories": {
                "data": "./data",
                "generated": "./data/generated",
                "training": "./data/training",
                "models": "./data/models"
            },
            "piper": {
                "executable_paths": ["piper"],
                "model_directories": ["./models"]
            },
            "training": {
                "default_architecture": "cnn",
                "learning_rate": 0.001,
                "batch_size": 32,
                "max_epochs": 100
            },
            "performance": {
                "cuda": {
                    "batch_size": 32,
                    "num_workers": 4
                }
            },
            "logging": {
                "level": "INFO",
                "file": "./logs/app.log"
            }
        }
    
    def get(self, *path, default=None) -> Any:
        """Get configuration value with dot notation support"""
        try:
            current = self._config
            for key in path:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self._config.get(section, {})
    
    def get_elevenlabs_config(self) -> Dict[str, Any]:
        """Get ElevenLabs configuration with API key from environment"""
        config = self.get_section('elevenlabs').copy()
        
        # Get API key from environment
        api_key_env = config.get('api_key_env', 'WAKEWORD_API_KEY')
        api_key = os.getenv(api_key_env)
        
        if api_key:
            config['api_key'] = api_key
        
        return config
    
    def get_piper_config(self) -> Dict[str, Any]:
        """Get Piper TTS configuration"""
        return self.get_section('piper')
    
    def get_augmentation_config(self) -> Dict[str, Any]:
        """Get augmentation configuration"""
        return self.get_section('augmentation')
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.get_section('training')
    
    def get_directories(self) -> Dict[str, str]:
        """Get all directory configurations"""
        return self.get_section('directories')
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.get_section('performance')
    
    def reload(self):
        """Reload configuration from file"""
        self._load_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary"""
        return self._config.copy()


# Global config instance
_config_instance = None


def get_config() -> UnifiedConfig:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = UnifiedConfig()
    return _config_instance


def reload_config():
    """Reload global configuration"""
    global _config_instance
    if _config_instance:
        _config_instance.reload()
    else:
        _config_instance = UnifiedConfig()