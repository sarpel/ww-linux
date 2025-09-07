"""
Augment Panel - Audio augmentation and dataset preparation interface
"""

import gradio as gr
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import tempfile
import os
import json

# Import local modules
from ..utils.archive_extractor import ArchiveExtractor
from ..utils.audio_converter import AudioConverter
from ..utils.file_manager import PanelCommunicator
from ..augmentation.audio_augmenter import AudioAugmenter

logger = logging.getLogger(__name__)


class AugmentPanel:
    """Augment panel controller for audio dataset preparation and augmentation"""
    
    def __init__(self, communicator: PanelCommunicator):
        self.communicator = communicator
        self.archive_extractor = ArchiveExtractor()
        self.audio_converter = AudioConverter()
        self.audio_augmenter = AudioAugmenter()
        
        # Setup directories
        self.input_dir = Path("data/augmented/input")
        self.positive_dir = Path("data/augmented/positive")
        self.negative_dir = Path("data/augmented/negative")
        self.processed_dir = Path("data/augmented/processed")
        
        for dir_path in [self.input_dir, self.positive_dir, self.negative_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def upload_and_extract_archive(
        self,
        archive_files: List[str],
        extract_to: str,
        progress=gr.Progress()
    ) -> Tuple[str, Dict]:
        """
        Upload and extract archive files to specified location
        
        Args:
            archive_files: List of uploaded archive file paths
            extract_to: Target extraction directory ('positive' or 'negative')
            progress: Gradio progress tracker
            
        Returns:
            Tuple[str, Dict]: Status message and extraction results
        """
        try:
            if not archive_files:
                return "<div class='error-panel'>❌ Error: No archive files provided</div>", {}
            
            progress(0, desc="Initializing extraction...")
            
            # Determine target directory
            if extract_to == "positive":
                target_dir = self.positive_dir
            elif extract_to == "negative":
                target_dir = self.negative_dir
            else:
                return f"<div class='error-panel'>❌ Error: Invalid extraction target: {extract_to}</div>", {}
            
            results = {
                'extracted_archives': 0,
                'total_archives': len(archive_files),
                'total_files_extracted': 0,
                'audio_files_found': 0,
                'failed_archives': []
            }
            
            for i, archive_file in enumerate(archive_files):
                progress((i + 1) / len(archive_files), desc=f"Extracting archive {i + 1}/{len(archive_files)}")
                
                archive_path = Path(archive_file)
                
                if not self.archive_extractor.is_supported_format(archive_path):
                    results['failed_archives'].append(f"{archive_path.name}: Unsupported format")
                    continue
                
                # Extract to subdirectory named after archive
                extract_target = target_dir / archive_path.stem
                
                def extraction_progress(current, total, filename):
                    # Update progress during extraction
                    base_progress = (i / len(archive_files))
                    current_progress = (current / total) * (1 / len(archive_files))
                    progress(base_progress + current_progress, desc=f"Extracting {filename[:30]}...")
                
                success = self.archive_extractor.extract_archive(
                    archive_path, extract_target, extraction_progress
                )
                
                if success:
                    results['extracted_archives'] += 1
                    
                    # Count extracted files
                    extracted_files = list(extract_target.rglob('*'))
                    extracted_files = [f for f in extracted_files if f.is_file()]
                    results['total_files_extracted'] += len(extracted_files)
                    
                    # Count audio files
                    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.pcm'}
                    audio_files = [f for f in extracted_files if f.suffix.lower() in audio_extensions]
                    results['audio_files_found'] += len(audio_files)
                    
                    # Convert non-WAV audio files to WAV
                    if audio_files:
                        self._convert_audio_files_to_wav(audio_files, progress)
                    
                else:
                    results['failed_archives'].append(f"{archive_path.name}: Extraction failed")
            
            # Generate status message
            if results['extracted_archives'] > 0:
                status = f"✅ Extracted {results['extracted_archives']}/{results['total_archives']} archives"
                status += f" ({results['audio_files_found']} audio files found)"
                
                if results['failed_archives']:
                    status += f" ⚠️ {len(results['failed_archives'])} failed"
            else:
                status = "❌ No archives were successfully extracted"
            
            return status, results
            
        except Exception as e:
            logger.error(f"Error in upload_and_extract_archive: {e}")
            return "<div class=\'error-panel\'>❌ Error: {str(e)}</div>", {}
    
    def _convert_audio_files_to_wav(self, audio_files: List[Path], progress):
        """Convert non-WAV audio files to WAV format"""
        try:
            non_wav_files = [f for f in audio_files if f.suffix.lower() != '.wav']
            
            if not non_wav_files:
                return
            
            progress(0.9, desc=f"Converting {len(non_wav_files)} audio files to WAV...")
            
            for audio_file in non_wav_files:
                output_path = audio_file.with_suffix('.wav')
                try:
                    self.audio_converter.convert_to_wav(audio_file, output_path)
                    # Remove original file after successful conversion
                    audio_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to convert {audio_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error converting audio files: {e}")
    
    def collect_all_audio_files(
        self,
        source_folder: str,
        target_category: str,
        progress=gr.Progress()
    ) -> Tuple[str, Dict]:
        """
        Collect all audio files from source folder (including subfolders) and move to target category
        
        Args:
            source_folder: Source folder to collect from ('input', 'positive', or 'negative')  
            target_category: Target category ('positive' or 'negative')
            progress: Gradio progress tracker
            
        Returns:
            Tuple[str, Dict]: Status message and collection results
        """
        try:
            progress(0, desc="Starting audio file collection...")
            
            # Get source directory
            if source_folder == "input":
                source_dir = self.input_dir
            elif source_folder == "positive":
                source_dir = self.positive_dir
            elif source_folder == "negative":  
                source_dir = self.negative_dir
            else:
                return "<div class='error-panel'>❌ Error: Invalid source folder: {source_folder}</div>", {}
            
            # Get target directory
            if target_category == "positive":
                target_dir = self.positive_dir
            elif target_category == "negative":
                target_dir = self.negative_dir
            else:
                return "<div class='error-panel'>❌ Error: Invalid target category: {target_category}</div>", {}
            
            # Find all audio files recursively (including subfolders)
            audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
            audio_files = []
            
            progress(0.1, desc="Scanning for audio files...")
            
            for ext in audio_extensions:
                audio_files.extend(source_dir.rglob(f'*{ext}'))
            
            if not audio_files:
                return "<div class='error-panel'>❌ Error: No audio files found in source folder</div>", {}
            
            progress(0.2, desc=f"Found {len(audio_files)} audio files, collecting...")
            
            results = {
                'total_files': len(audio_files),
                'moved_files': 0,
                'errors': [],
                'skipped_same_location': 0
            }
            
            for i, audio_file in enumerate(audio_files):
                progress(0.2 + (0.8 * (i + 1) / len(audio_files)), 
                        desc=f"Moving file {i + 1}/{len(audio_files)}")
                
                try:
                    # Skip if file is already in target directory
                    if audio_file.parent == target_dir:
                        results['skipped_same_location'] += 1
                        continue
                    
                    # Generate target file path
                    target_file = target_dir / audio_file.name
                    
                    # Handle filename conflicts by adding counter
                    if target_file.exists():
                        counter = 1
                        stem = audio_file.stem
                        suffix = audio_file.suffix
                        while target_file.exists():
                            target_file = target_dir / f"{stem}_{counter:04d}{suffix}"
                            counter += 1
                    
                    # Move file to target directory
                    audio_file.rename(target_file)
                    results['moved_files'] += 1
                    
                except Exception as e:
                    results['errors'].append(f"{audio_file.name}: {str(e)}")
                    logger.warning(f"Failed to move {audio_file}: {e}")
            
            # Generate status message
            status = f"<div class='success-panel'>✅ Collection complete: {results['moved_files']} files moved to {target_category}"
            
            if results['skipped_same_location'] > 0:
                status += f", {results['skipped_same_location']} already in target"
            
            if results['errors']:
                status += f", {len(results['errors'])} errors"
            
            status += "</div>"
            
            return status, results
            
        except Exception as e:
            logger.error(f"Error in collect_all_audio_files: {e}")
            return "<div class='error-panel'>❌ Error: {str(e)}</div>", {}
    
    def run_audio_augmentation(
        self,
        target_category: str,
        augmentation_config: str,
        num_augmentations: int,
        progress=gr.Progress()
    ) -> Tuple[str, Dict]:
        """
        Run audio augmentation on selected category
        
        Args:
            target_category: Category to augment ('positive' or 'negative')
            augmentation_config: JSON configuration for augmentation
            num_augmentations: Number of augmented versions per original
            progress: Gradio progress tracker
            
        Returns:
            Tuple[str, Dict]: Status message and augmentation results
        """
        try:
            progress(0, desc="Initializing augmentation...")
            
            # Parse augmentation config
            try:
                config = json.loads(augmentation_config)
            except json.JSONDecodeError:
                return "<div class=\'error-panel\'>❌ Error: Invalid augmentation configuration JSON</div>", {}
            
            # Get source directory
            if target_category == "positive":
                source_dir = self.positive_dir
            elif target_category == "negative":
                source_dir = self.negative_dir
            else:
                return "<div class=\'error-panel\'>❌ Error: Invalid target category: {target_category}</div>", {}
            
            # Find original audio files
            audio_files = list(source_dir.glob('*.wav'))
            
            if not audio_files:
                return "<div class=\'error-panel\'>❌ Error: No WAV files found in {target_category} folder</div>", {}
            
            progress(0.1, desc=f"Found {len(audio_files)} files for augmentation")
            
            # Run augmentation
            results = self.audio_augmenter.batch_augment(
                source_dir,
                source_dir,  # Output to same directory
                config,
                num_augmentations,
                lambda p, d: progress(0.1 + 0.9 * p, desc=d)
            )
            
            # Generate status message
            if results.get('successful', 0) > 0:
                status = f"✅ Augmented {results['successful']} files"
                status += f" (generated {results.get('total_generated', 0)} variations)"
                
                if results.get('failed', 0) > 0:
                    status += f" ⚠️ {results['failed']} failed"
            else:
                status = "❌ Augmentation failed"
            
            return status, results
            
        except Exception as e:
            logger.error(f"Error in run_audio_augmentation: {e}")
            return "<div class=\'error-panel\'>❌ Error: {str(e)}</div>", {}
    
    def preview_augmented_sample(
        self,
        category: str,
        file_index: int,
        augmentation_type: str
    ) -> Optional[str]:
        """Preview an augmented audio sample"""
        try:
            if category == "positive":
                source_dir = self.positive_dir
            elif category == "negative":
                source_dir = self.negative_dir
            else:
                return None
            
            # Find files matching augmentation type
            pattern = f"*_{augmentation_type}_*.wav"
            audio_files = list(source_dir.glob(pattern))
            
            if not audio_files or file_index >= len(audio_files):
                return None
            
            return str(audio_files[file_index])
            
        except Exception as e:
            logger.error(f"Error in preview_augmented_sample: {e}")
            return None
    
    def get_augmentation_stats(self) -> str:
        """Get statistics about augmented files"""
        try:
            positive_stats = self.communicator.file_manager.get_folder_stats(self.positive_dir)
            negative_stats = self.communicator.file_manager.get_folder_stats(self.negative_dir)
            input_stats = self.communicator.file_manager.get_folder_stats(self.input_dir)
            
            return f"""
            **Augmentation Statistics:**
            
            📁 **Input Folder:**
            - Audio files: {input_stats.get('audio_files', 0)}
            - Total size: {input_stats.get('folder_size_mb', 0):.1f} MB
            
            ✅ **Positive Samples:**
            - Audio files: {positive_stats.get('audio_files', 0)}
            - Total size: {positive_stats.get('folder_size_mb', 0):.1f} MB
            
            ❌ **Negative Samples:**
            - Audio files: {negative_stats.get('audio_files', 0)}
            - Total size: {negative_stats.get('folder_size_mb', 0):.1f} MB
            
            **Last Modified:** {positive_stats.get('last_modified') or 'Unknown'}
            """
            
        except Exception as e:
            logger.error(f"Error getting augmentation stats: {e}")
            return "Error getting statistics"
    
    def prepare_for_training(self, progress=gr.Progress()) -> str:
        """Prepare augmented data for training pipeline"""
        try:
            progress(0, desc="Preparing data for training...")
            
            # Validate that we have both positive and negative samples
            pos_stats = self.communicator.file_manager.get_folder_stats(self.positive_dir)
            neg_stats = self.communicator.file_manager.get_folder_stats(self.negative_dir)
            
            if pos_stats.get('audio_files', 0) == 0:
                return "<div class=\'error-panel\'>❌ Error: No positive samples available for training</div>"
            
            if neg_stats.get('audio_files', 0) == 0:
                return "<div class=\'error-panel\'>❌ Error: No negative samples available for training</div>"
            
            progress(0.5, desc="Triggering transfer to training panel...")
            
            # Trigger transfer to training panel
            success = self.communicator.trigger_augment_to_train()
            
            if success:
                return "<div class=\'success-panel\'>✅ Data prepared for training ({pos_stats['audio_files']} positive, {neg_stats['audio_files']} negative samples)</div>"
            else:
                return "<div class=\'warning-panel\'>⚠️ Data preparation completed but transfer to training failed</div>"
                
        except Exception as e:
            logger.error(f"Error preparing for training: {e}")
            return "<div class=\'error-panel\'>❌ Error: {str(e)}</div>"
    
    def clear_category(self, category: str) -> str:
        """Clear all files in specified category"""
        try:
            if category == "positive":
                target_dir = self.positive_dir
            elif category == "negative":
                target_dir = self.negative_dir
            elif category == "input":
                target_dir = self.input_dir
            else:
                return "<div class=\'error-panel\'>❌ Error: Invalid category: {category}</div>"
            
            removed, total = self.communicator.file_manager.clean_directory(
                target_dir,
                file_patterns=["*.*"]
            )
            
            if removed > 0:
                return "<div class=\'success-panel\'>✅ Cleared {removed} files from {category} folder</div>"
            else:
                return f"No files to clear in {category} folder"
                
        except Exception as e:
            logger.error(f"Error clearing category: {e}")
            return "<div class=\'error-panel\'>❌ Error: {str(e)}</div>"


def create_augment_panel(communicator: PanelCommunicator):
    """Create the augment panel UI components"""
    
    panel = AugmentPanel(communicator)
    
    with gr.Column(elem_classes=["panel-container"]):
        
        # Archive upload section
        with gr.Group():
            gr.Markdown("### 📦 Archive Upload & Extraction")
            
            with gr.Row():
                archive_upload = gr.File(
                    label="Upload Archive Files",
                    file_count="multiple",
                    file_types=[".zip", ".7z", ".rar", ".tar", ".tar.gz", ".tar.bz2", ".tar.xz"]
                )
                
                extract_target = gr.Dropdown(
                    label="Extract to Category",
                    choices=["positive", "negative"],
                    value="positive"
                )
            
            extract_btn = gr.Button("🚀 Extract Archives", variant="primary")
            extraction_status = gr.HTML(value="Ready to extract archives")
        
        # Audio file collection section
        with gr.Group():
            gr.Markdown("### 📂 Audio File Collection")
            gr.Markdown("**Collect all audio files from a folder (including subfolders) and move to target category**")
            
            with gr.Row():
                collect_source = gr.Dropdown(
                    label="Source Folder",
                    choices=["input", "positive", "negative"],
                    value="input",
                    info="Folder to collect audio files from"
                )
                
                collect_target = gr.Dropdown(
                    label="Target Category",
                    choices=["positive", "negative"],
                    value="positive",
                    info="Where to move the collected files"
                )
            
            collect_btn = gr.Button("📁 Collect All Audio Files", variant="secondary", size="lg")
            collect_status = gr.HTML(value="<div class='info-panel'>📝 Ready to collect audio files from folders and subfolders</div>")
        
        # Audio augmentation section
        with gr.Group():
            gr.Markdown("### 🎛️ Audio Augmentation")
            
            with gr.Row():
                augment_category = gr.Dropdown(
                    label="Category to Augment",
                    choices=["positive", "negative"],
                    value="positive"
                )
                
                num_augmentations = gr.Slider(
                    label="Augmentations per File",
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1
                )
            
            augmentation_config = gr.Code(
                label="Augmentation Configuration (JSON)",
                language="json",
                value="""{
  "noise": {"enabled": true, "intensity": 0.1},
  "pitch_shift": {"enabled": true, "semitones_range": [-2, 2]},
  "speed_change": {"enabled": true, "factor_range": [0.9, 1.1]},
  "volume_change": {"enabled": true, "factor_range": [0.8, 1.2]}
}"""
            )
            
            augment_btn = gr.Button("🎵 Run Augmentation", variant="primary")
            augmentation_status = gr.HTML(value="Ready to run augmentation")
        
        # Preview section
        with gr.Group():
            gr.Markdown("### 👁️ Sample Preview")
            
            with gr.Row():
                preview_category = gr.Dropdown(
                    label="Preview Category",
                    choices=["positive", "negative"],
                    value="positive"
                )
                
                augmentation_type = gr.Dropdown(
                    label="Augmentation Type",
                    choices=["original", "noise", "pitch", "speed", "volume"],
                    value="original"
                )
                
                sample_index = gr.Slider(
                    label="Sample Index",
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1
                )
            
            preview_audio = gr.Audio(
                label="Audio Preview",
                interactive=False
            )
        
        # Statistics and management
        with gr.Group():
            gr.Markdown("### 📊 Statistics & Management")
            
            stats_display = gr.HTML(
                value=panel.get_augmentation_stats()
            )
            
            with gr.Row():
                refresh_stats_btn = gr.Button("🔄 Refresh Stats", variant="secondary")
                prepare_training_btn = gr.Button("🚀 Prepare for Training", variant="primary")
            
            with gr.Row():
                clear_positive_btn = gr.Button("🗑️ Clear Positive", variant="stop")
                clear_negative_btn = gr.Button("🗑️ Clear Negative", variant="stop")
                clear_input_btn = gr.Button("🗑️ Clear Input", variant="stop")
        
        # Event handlers
        def on_extract_archives(archive_files, extract_to, progress=gr.Progress()):
            if not archive_files:
                return "<div class=\'error-panel\'>❌ No files uploaded</div>"
            
            return panel.upload_and_extract_archive(
                [f.name for f in archive_files], extract_to, progress
            )[0]  # Return only status message
        
        def on_collect_audio_files(source, target, progress=gr.Progress()):
            return panel.collect_all_audio_files(
                source, target, progress
            )[0]  # Return only status message
        
        def on_run_augmentation(category, config, num_augs, progress=gr.Progress()):
            return panel.run_audio_augmentation(
                category, config, num_augs, progress
            )[0]  # Return only status message
        
        def on_preview_sample(category, aug_type, index):
            return panel.preview_augmented_sample(category, int(index), aug_type)
        
        def on_refresh_stats():
            return panel.get_augmentation_stats()
        
        def on_prepare_training(progress=gr.Progress()):
            return panel.prepare_for_training(progress)
        
        def on_clear_category(category):
            return panel.clear_category(category)
        
        # Connect events
        extract_btn.click(
            fn=on_extract_archives,
            inputs=[archive_upload, extract_target],
            outputs=[extraction_status]
        )
        
        collect_btn.click(
            fn=on_collect_audio_files,
            inputs=[collect_source, collect_target],
            outputs=[collect_status]
        )
        
        augment_btn.click(
            fn=on_run_augmentation,
            inputs=[augment_category, augmentation_config, num_augmentations],
            outputs=[augmentation_status]
        )
        
        preview_category.change(
            fn=on_preview_sample,
            inputs=[preview_category, augmentation_type, sample_index],
            outputs=[preview_audio]
        )
        
        augmentation_type.change(
            fn=on_preview_sample,
            inputs=[preview_category, augmentation_type, sample_index],
            outputs=[preview_audio]
        )
        
        sample_index.change(
            fn=on_preview_sample,
            inputs=[preview_category, augmentation_type, sample_index],
            outputs=[preview_audio]
        )
        
        refresh_stats_btn.click(
            fn=on_refresh_stats,
            outputs=[stats_display]
        )
        
        prepare_training_btn.click(
            fn=on_prepare_training,
            outputs=[extraction_status]  # Reuse extraction status for training prep
        )
        
        clear_positive_btn.click(
            fn=lambda: on_clear_category("positive"),
            outputs=[collect_status]
        )
        
        clear_negative_btn.click(
            fn=lambda: on_clear_category("negative"),
            outputs=[collect_status]
        )
        
        clear_input_btn.click(
            fn=lambda: on_clear_category("input"),
            outputs=[collect_status]
        )
    
    return {
        'archive_upload': archive_upload,
        'extract_target': extract_target,
        'extract_btn': extract_btn,
        'extraction_status': extraction_status,
        'collect_source': collect_source,
        'collect_target': collect_target,
        'collect_btn': collect_btn,
        'collect_status': collect_status,
        'augment_category': augment_category,
        'num_augmentations': num_augmentations,
        'augmentation_config': augmentation_config,
        'augment_btn': augment_btn,
        'augmentation_status': augmentation_status,
        'preview_category': preview_category,
        'augmentation_type': augmentation_type,
        'sample_index': sample_index,
        'preview_audio': preview_audio,
        'stats_display': stats_display,
        'prepare_training_btn': prepare_training_btn
    }