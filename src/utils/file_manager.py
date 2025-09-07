"""
Cross-panel file operations and folder management utilities
"""

import os
import shutil
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class FileManager:
    """Cross-panel file operations and state management"""
    
    def __init__(self, base_dir: Union[str, Path] = "data"):
        self.base_dir = Path(base_dir)
        self.state_file = self.base_dir / "file_manager_state.json"
        self._ensure_base_structure()
        
    def _ensure_base_structure(self):
        """Ensure base directory structure exists"""
        directories = [
            "generated",
            "augmented/input",
            "augmented/positive", 
            "augmented/negative",
            "augmented/processed",
            "training",
            "models"
        ]
        
        for directory in directories:
            (self.base_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def get_folder_stats(self, folder_path: Union[str, Path]) -> Dict:
        """
        Get statistics for a folder
        
        Args:
            folder_path: Path to folder
            
        Returns:
            Dict: Folder statistics
        """
        try:
            folder_path = Path(folder_path)
            
            if not folder_path.exists():
                return {
                    'exists': False,
                    'file_count': 0,
                    'total_size': 0,
                    'audio_files': 0,
                    'last_modified': None
                }
            
            files = list(folder_path.rglob('*'))
            files = [f for f in files if f.is_file()]
            
            # Audio file extensions
            audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.pcm'}
            audio_files = [f for f in files if f.suffix.lower() in audio_extensions]
            
            total_size = sum(f.stat().st_size for f in files)
            
            # Get most recent modification time
            last_modified = None
            if files:
                last_modified_timestamp = max(f.stat().st_mtime for f in files)
                last_modified = datetime.fromtimestamp(last_modified_timestamp)
            
            return {
                'exists': True,
                'file_count': len(files),
                'total_size': total_size,
                'audio_files': len(audio_files),
                'audio_file_list': [f.name for f in audio_files[:10]],  # First 10 files
                'last_modified': last_modified.isoformat() if last_modified else None,
                'folder_size_mb': total_size / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Error getting folder stats for {folder_path}: {e}")
            return {'error': str(e)}
    
    def move_files_between_panels(
        self,
        source_dir: Union[str, Path],
        target_dir: Union[str, Path],
        file_pattern: str = "*",
        copy_instead: bool = False
    ) -> Tuple[int, int, List[str]]:
        """
        Move or copy files between panel directories
        
        Args:
            source_dir: Source directory
            target_dir: Target directory
            file_pattern: File pattern to match (default: all files)
            copy_instead: Copy instead of move if True
            
        Returns:
            Tuple[int, int, List[str]]: (successful, total, failed_files)
        """
        try:
            source_dir = Path(source_dir)
            target_dir = Path(target_dir)
            
            if not source_dir.exists():
                logger.error(f"Source directory not found: {source_dir}")
                return 0, 0, []
            
            # Ensure target directory exists
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Find matching files
            files = list(source_dir.glob(file_pattern))
            files = [f for f in files if f.is_file()]
            
            successful = 0
            failed_files = []
            
            for file_path in files:
                try:
                    target_file = target_dir / file_path.name
                    
                    # Handle file conflicts
                    if target_file.exists():
                        target_file = self._get_unique_filename(target_file)
                    
                    if copy_instead:
                        shutil.copy2(file_path, target_file)
                    else:
                        shutil.move(str(file_path), str(target_file))
                    
                    successful += 1
                    logger.debug(f"{'Copied' if copy_instead else 'Moved'} {file_path} to {target_file}")
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    failed_files.append(file_path.name)
            
            operation = "copy" if copy_instead else "move"
            logger.info(f"File {operation} complete: {successful}/{len(files)} files processed")
            
            return successful, len(files), failed_files
            
        except Exception as e:
            logger.error(f"Error in move_files_between_panels: {e}")
            return 0, 0, []
    
    def _get_unique_filename(self, file_path: Path) -> Path:
        """Generate unique filename if file already exists"""
        counter = 1
        stem = file_path.stem
        suffix = file_path.suffix
        parent = file_path.parent
        
        while file_path.exists():
            new_name = f"{stem}_{counter}{suffix}"
            file_path = parent / new_name
            counter += 1
        
        return file_path
    
    def clean_directory(
        self,
        directory: Union[str, Path],
        older_than_days: Optional[int] = None,
        file_patterns: Optional[List[str]] = None
    ) -> Tuple[int, int]:
        """
        Clean directory of old or specific files
        
        Args:
            directory: Directory to clean
            older_than_days: Remove files older than N days (optional)
            file_patterns: List of file patterns to remove (optional)
            
        Returns:
            Tuple[int, int]: (files_removed, total_files)
        """
        try:
            directory = Path(directory)
            
            if not directory.exists():
                return 0, 0
            
            files_to_remove = []
            
            if older_than_days is not None:
                cutoff_time = datetime.now().timestamp() - (older_than_days * 24 * 3600)
                old_files = [f for f in directory.rglob('*') if f.is_file() and f.stat().st_mtime < cutoff_time]
                files_to_remove.extend(old_files)
            
            if file_patterns:
                for pattern in file_patterns:
                    pattern_files = list(directory.glob(pattern))
                    files_to_remove.extend([f for f in pattern_files if f.is_file()])
            
            # Remove duplicates
            files_to_remove = list(set(files_to_remove))
            
            removed_count = 0
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Could not remove file {file_path}: {e}")
            
            logger.info(f"Directory cleanup: removed {removed_count}/{len(files_to_remove)} files")
            return removed_count, len(files_to_remove)
            
        except Exception as e:
            logger.error(f"Error cleaning directory {directory}: {e}")
            return 0, 0
    
    def save_state(self, state_data: Dict):
        """
        Save file manager state to disk
        
        Args:
            state_data: State data to save
        """
        try:
            state_data['timestamp'] = datetime.now().isoformat()
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.debug("File manager state saved")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def load_state(self) -> Dict:
        """
        Load file manager state from disk
        
        Returns:
            Dict: Loaded state data
        """
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            
            return {}
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return {}
    
    def get_panel_status(self) -> Dict[str, Dict]:
        """
        Get status of all panels based on file presence
        
        Returns:
            Dict[str, Dict]: Status of each panel
        """
        panels = {
            'generate': self.base_dir / 'generated',
            'augment_input': self.base_dir / 'augmented' / 'input',
            'augment_positive': self.base_dir / 'augmented' / 'positive',
            'augment_negative': self.base_dir / 'augmented' / 'negative',
            'augment_processed': self.base_dir / 'augmented' / 'processed',
            'training': self.base_dir / 'training',
            'models': self.base_dir / 'models'
        }
        
        status = {}
        for panel_name, panel_path in panels.items():
            status[panel_name] = self.get_folder_stats(panel_path)
        
        return status
    
    def setup_training_data(
        self,
        positive_source: Union[str, Path],
        negative_source: Union[str, Path],
        output_dir: Union[str, Path],
        train_split: float = 0.8
    ) -> Dict:
        """
        Setup training data structure from positive/negative samples
        
        Args:
            positive_source: Directory with positive samples
            negative_source: Directory with negative samples
            output_dir: Output directory for training structure
            train_split: Fraction for training set
            
        Returns:
            Dict: Setup results and statistics
        """
        try:
            positive_source = Path(positive_source)
            negative_source = Path(negative_source)
            output_dir = Path(output_dir)
            
            # Create training structure
            train_pos = output_dir / 'train' / 'positive'
            train_neg = output_dir / 'train' / 'negative'
            val_pos = output_dir / 'validation' / 'positive'
            val_neg = output_dir / 'validation' / 'negative'
            
            for dir_path in [train_pos, train_neg, val_pos, val_neg]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            results = {
                'positive': {'train': 0, 'validation': 0},
                'negative': {'train': 0, 'validation': 0}
            }
            
            # Process positive samples
            if positive_source.exists():
                pos_files = list(positive_source.glob('*.wav'))
                train_count = int(len(pos_files) * train_split)
                
                for i, file_path in enumerate(pos_files):
                    target_dir = train_pos if i < train_count else val_pos
                    shutil.copy2(file_path, target_dir / file_path.name)
                    
                    if i < train_count:
                        results['positive']['train'] += 1
                    else:
                        results['positive']['validation'] += 1
            
            # Process negative samples
            if negative_source.exists():
                neg_files = list(negative_source.glob('*.wav'))
                train_count = int(len(neg_files) * train_split)
                
                for i, file_path in enumerate(neg_files):
                    target_dir = train_neg if i < train_count else val_neg
                    shutil.copy2(file_path, target_dir / file_path.name)
                    
                    if i < train_count:
                        results['negative']['train'] += 1
                    else:
                        results['negative']['validation'] += 1
            
            results['train_split'] = train_split
            results['output_dir'] = str(output_dir)
            
            logger.info(f"Training data setup complete: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error setting up training data: {e}")
            return {'error': str(e)}
    
    def validate_audio_files(self, directory: Union[str, Path]) -> Dict:
        """
        Validate audio files in directory
        
        Args:
            directory: Directory to validate
            
        Returns:
            Dict: Validation results
        """
        try:
            directory = Path(directory)
            
            if not directory.exists():
                return {'error': 'Directory does not exist'}
            
            audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.pcm'}
            audio_files = []
            
            for ext in audio_extensions:
                audio_files.extend(directory.glob(f'*{ext}'))
            
            valid_files = []
            invalid_files = []
            
            for file_path in audio_files:
                try:
                    # Basic validation - check if file can be opened and has size > 0
                    if file_path.stat().st_size > 0:
                        valid_files.append(file_path.name)
                    else:
                        invalid_files.append(file_path.name)
                except:
                    invalid_files.append(file_path.name)
            
            return {
                'total_files': len(audio_files),
                'valid_files': len(valid_files),
                'invalid_files': len(invalid_files),
                'invalid_file_list': invalid_files,
                'validation_success': len(invalid_files) == 0
            }
            
        except Exception as e:
            logger.error(f"Error validating audio files: {e}")
            return {'error': str(e)}


class PanelCommunicator:
    """Handle communication and data flow between panels"""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
        self.callbacks = {}
    
    def register_callback(self, event_type: str, callback):
        """Register callback for panel events"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
    
    def notify_panels(self, event_type: str, data: Dict):
        """Notify registered panels of events"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in panel callback: {e}")
    
    def trigger_generate_to_augment(self) -> bool:
        """Trigger file transfer from generate to augment panel"""
        try:
            source_dir = self.file_manager.base_dir / 'generated'
            target_dir = self.file_manager.base_dir / 'augmented' / 'input'
            
            # Convert any PCM files to WAV first
            from .audio_converter import AudioConverter
            converter = AudioConverter()
            converter.batch_convert_pcm_to_wav(source_dir, source_dir)
            
            # Move WAV files to augment input
            success, total, failed = self.file_manager.move_files_between_panels(
                source_dir, target_dir, "*.wav", copy_instead=True
            )
            
            # Notify augment panel
            self.notify_panels('files_transferred', {
                'from': 'generate',
                'to': 'augment',
                'success': success,
                'total': total,
                'failed': failed
            })
            
            return success > 0
            
        except Exception as e:
            logger.error(f"Error in generate to augment transfer: {e}")
            return False
    
    def trigger_augment_to_train(self) -> bool:
        """Trigger file transfer from augment to train panel"""
        try:
            source_dir = self.file_manager.base_dir / 'augmented' / 'processed'
            target_dir = self.file_manager.base_dir / 'training'
            
            # Setup training data structure
            results = self.file_manager.setup_training_data(
                self.file_manager.base_dir / 'augmented' / 'positive',
                self.file_manager.base_dir / 'augmented' / 'negative',
                target_dir
            )
            
            # Notify train panel
            self.notify_panels('training_data_ready', {
                'from': 'augment',
                'to': 'train',
                'results': results
            })
            
            return 'error' not in results
            
        except Exception as e:
            logger.error(f"Error in augment to train transfer: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    file_manager = FileManager()
    communicator = PanelCommunicator(file_manager)
    
    # Get panel status
    status = file_manager.get_panel_status()
    print("Panel status:", status)
    
    print("File manager utilities ready!")