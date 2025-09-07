"""
Multi-format archive extraction utilities for compressed dataset handling
"""

import os
import zipfile
import tarfile
import py7zr
import rarfile
from pathlib import Path
from typing import Union, List, Callable, Optional, Tuple
import logging
from tqdm import tqdm
import shutil

logger = logging.getLogger(__name__)


class ArchiveExtractor:
    """Multi-format archive extraction with progress tracking"""
    
    def __init__(self):
        self.supported_formats = ['.7z', '.zip', '.rar', '.tar', '.tar.gz', '.tar.bz2', '.tar.xz']
        self._setup_rarfile()
    
    def _setup_rarfile(self):
        """Setup rarfile with appropriate unrar tool"""
        try:
            # Try to find unrar executable
            unrar_path = shutil.which('unrar')
            if unrar_path:
                rarfile.UNRAR_TOOL = unrar_path
            else:
                # Try alternative paths
                alternative_paths = [
                    r'C:\Program Files\WinRAR\UnRAR.exe',
                    r'C:\Program Files (x86)\WinRAR\UnRAR.exe',
                    '/usr/bin/unrar',
                    '/usr/local/bin/unrar'
                ]
                
                for path in alternative_paths:
                    if os.path.exists(path):
                        rarfile.UNRAR_TOOL = path
                        break
                else:
                    logger.info("UnRAR tool not found. RAR extraction disabled.")
                    
        except Exception as e:
            logger.warning(f"Error setting up RAR support: {e}")
    
    def get_archive_type(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Determine archive type from file extension
        
        Args:
            file_path: Path to archive file
            
        Returns:
            str: Archive type or None if unsupported
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix == '.7z':
            return '7z'
        elif suffix == '.zip':
            return 'zip'
        elif suffix == '.rar':
            return 'rar'
        elif suffix == '.tar':
            return 'tar'
        elif file_path.name.endswith('.tar.gz'):
            return 'tar.gz'
        elif file_path.name.endswith('.tar.bz2'):
            return 'tar.bz2'
        elif file_path.name.endswith('.tar.xz'):
            return 'tar.xz'
        
        return None
    
    def is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """
        Check if archive format is supported
        
        Args:
            file_path: Path to archive file
            
        Returns:
            bool: True if supported, False otherwise
        """
        return self.get_archive_type(file_path) is not None
    
    def extract_archive(
        self,
        file_path: Union[str, Path],
        target_dir: Union[str, Path],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> bool:
        """
        Extract archive to target directory with progress callback
        
        Args:
            file_path: Path to archive file
            target_dir: Target extraction directory
            progress_callback: Optional callback for progress updates (current, total, filename)
            
        Returns:
            bool: True if extraction successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            target_dir = Path(target_dir)
            
            if not file_path.exists():
                logger.error(f"Archive file not found: {file_path}")
                return False
            
            # Create target directory
            target_dir.mkdir(parents=True, exist_ok=True)
            
            archive_type = self.get_archive_type(file_path)
            if not archive_type:
                logger.error(f"Unsupported archive format: {file_path}")
                return False
            
            logger.info(f"Extracting {archive_type} archive: {file_path}")
            
            # Extract based on archive type
            if archive_type == 'zip':
                return self._extract_zip(file_path, target_dir, progress_callback)
            elif archive_type == '7z':
                return self._extract_7z(file_path, target_dir, progress_callback)
            elif archive_type == 'rar':
                return self._extract_rar(file_path, target_dir, progress_callback)
            elif archive_type in ['tar', 'tar.gz', 'tar.bz2', 'tar.xz']:
                return self._extract_tar(file_path, target_dir, progress_callback, archive_type)
            
            return False
            
        except Exception as e:
            logger.error(f"Error extracting archive {file_path}: {e}")
            return False
    
    def _extract_zip(
        self,
        file_path: Path,
        target_dir: Path,
        progress_callback: Optional[Callable[[int, int, str], None]]
    ) -> bool:
        """Extract ZIP archive"""
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                members = zip_ref.namelist()
                total_files = len(members)
                
                for i, member in enumerate(members):
                    zip_ref.extract(member, target_dir)
                    
                    if progress_callback:
                        progress_callback(i + 1, total_files, member)
                
                logger.info(f"Successfully extracted {total_files} files from ZIP archive")
                return True
                
        except Exception as e:
            logger.error(f"Error extracting ZIP archive: {e}")
            return False
    
    def _extract_7z(
        self,
        file_path: Path,
        target_dir: Path,
        progress_callback: Optional[Callable[[int, int, str], None]]
    ) -> bool:
        """Extract 7z archive"""
        try:
            with py7zr.SevenZipFile(file_path, 'r') as archive:
                members = archive.getnames()
                total_files = len(members)
                
                archive.extractall(path=target_dir)
                
                # Since py7zr doesn't provide per-file extraction callback,
                # we simulate progress updates
                if progress_callback:
                    for i, member in enumerate(members):
                        progress_callback(i + 1, total_files, member)
                
                logger.info(f"Successfully extracted {total_files} files from 7z archive")
                return True
                
        except Exception as e:
            logger.error(f"Error extracting 7z archive: {e}")
            return False
    
    def _extract_rar(
        self,
        file_path: Path,
        target_dir: Path,
        progress_callback: Optional[Callable[[int, int, str], None]]
    ) -> bool:
        """Extract RAR archive"""
        try:
            with rarfile.RarFile(file_path, 'r') as rar_ref:
                members = rar_ref.namelist()
                total_files = len(members)
                
                for i, member in enumerate(members):
                    rar_ref.extract(member, target_dir)
                    
                    if progress_callback:
                        progress_callback(i + 1, total_files, member)
                
                logger.info(f"Successfully extracted {total_files} files from RAR archive")
                return True
                
        except Exception as e:
            logger.error(f"Error extracting RAR archive: {e}")
            return False
    
    def _extract_tar(
        self,
        file_path: Path,
        target_dir: Path,
        progress_callback: Optional[Callable[[int, int, str], None]],
        archive_type: str
    ) -> bool:
        """Extract TAR archive (including compressed variants)"""
        try:
            # Determine open mode based on archive type
            mode_map = {
                'tar': 'r',
                'tar.gz': 'r:gz',
                'tar.bz2': 'r:bz2',
                'tar.xz': 'r:xz'
            }
            
            mode = mode_map.get(archive_type, 'r')
            
            with tarfile.open(file_path, mode) as tar_ref:
                members = tar_ref.getnames()
                total_files = len(members)
                
                for i, member in enumerate(members):
                    tar_ref.extract(member, target_dir)
                    
                    if progress_callback:
                        progress_callback(i + 1, total_files, member)
                
                logger.info(f"Successfully extracted {total_files} files from TAR archive")
                return True
                
        except Exception as e:
            logger.error(f"Error extracting TAR archive: {e}")
            return False
    
    def list_archive_contents(self, file_path: Union[str, Path]) -> List[str]:
        """
        List contents of archive without extracting
        
        Args:
            file_path: Path to archive file
            
        Returns:
            List[str]: List of file names in archive
        """
        try:
            file_path = Path(file_path)
            archive_type = self.get_archive_type(file_path)
            
            if not archive_type:
                return []
            
            if archive_type == 'zip':
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    return zip_ref.namelist()
            
            elif archive_type == '7z':
                with py7zr.SevenZipFile(file_path, 'r') as archive:
                    return archive.getnames()
            
            elif archive_type == 'rar':
                with rarfile.RarFile(file_path, 'r') as rar_ref:
                    return rar_ref.namelist()
            
            elif archive_type in ['tar', 'tar.gz', 'tar.bz2', 'tar.xz']:
                mode_map = {
                    'tar': 'r',
                    'tar.gz': 'r:gz',
                    'tar.bz2': 'r:bz2',
                    'tar.xz': 'r:xz'
                }
                mode = mode_map.get(archive_type, 'r')
                
                with tarfile.open(file_path, mode) as tar_ref:
                    return tar_ref.getnames()
            
            return []
            
        except Exception as e:
            logger.error(f"Error listing archive contents: {e}")
            return []
    
    def get_archive_info(self, file_path: Union[str, Path]) -> dict:
        """
        Get information about archive
        
        Args:
            file_path: Path to archive file
            
        Returns:
            dict: Archive information
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {}
            
            contents = self.list_archive_contents(file_path)
            
            # Count audio files
            audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.pcm'}
            audio_files = [f for f in contents if Path(f).suffix.lower() in audio_extensions]
            
            return {
                'file_path': str(file_path),
                'archive_type': self.get_archive_type(file_path),
                'file_size': file_path.stat().st_size,
                'total_files': len(contents),
                'audio_files': len(audio_files),
                'file_list': contents[:50],  # Show first 50 files
                'audio_file_list': audio_files[:20]  # Show first 20 audio files
            }
            
        except Exception as e:
            logger.error(f"Error getting archive info: {e}")
            return {}
    
    def batch_extract_archives(
        self,
        archive_dir: Union[str, Path],
        target_base_dir: Union[str, Path],
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Tuple[int, int, List[str]]:
        """
        Extract all supported archives in directory
        
        Args:
            archive_dir: Directory containing archives
            target_base_dir: Base directory for extraction
            progress_callback: Optional callback for progress (archive_name, current, total)
            
        Returns:
            Tuple[int, int, List[str]]: (successful, total, failed_files)
        """
        archive_dir = Path(archive_dir)
        target_base_dir = Path(target_base_dir)
        
        if not archive_dir.exists():
            logger.error(f"Archive directory not found: {archive_dir}")
            return 0, 0, []
        
        # Find all supported archives
        archives = []
        for format_ext in self.supported_formats:
            if format_ext.startswith('.tar.'):
                # Handle compound extensions
                pattern = f"*{format_ext}"
            else:
                pattern = f"*{format_ext}"
            archives.extend(archive_dir.glob(pattern))
        
        successful = 0
        failed_files = []
        total_archives = len(archives)
        
        for i, archive_path in enumerate(archives):
            archive_name = archive_path.name
            target_dir = target_base_dir / archive_path.stem
            
            if progress_callback:
                progress_callback(archive_name, i + 1, total_archives)
            
            if self.extract_archive(archive_path, target_dir):
                successful += 1
            else:
                failed_files.append(archive_name)
        
        logger.info(f"Batch extraction complete: {successful}/{total_archives} archives extracted")
        return successful, total_archives, failed_files


def create_progress_callback(use_tqdm: bool = True):
    """
    Create a progress callback function
    
    Args:
        use_tqdm: Whether to use tqdm for progress display
        
    Returns:
        Callable: Progress callback function
    """
    if use_tqdm:
        pbar = None
        
        def callback(current: int, total: int, filename: str):
            nonlocal pbar
            if pbar is None:
                pbar = tqdm(total=total, desc="Extracting")
            pbar.update(1)
            pbar.set_postfix(file=filename[:30] + "..." if len(filename) > 30 else filename)
            
            if current >= total:
                pbar.close()
                pbar = None
        
        return callback
    else:
        def callback(current: int, total: int, filename: str):
            percent = (current / total) * 100 if total > 0 else 0
            print(f"\rExtracting: {percent:.1f}% ({current}/{total}) - {filename[:30]}...", end="")
            
            if current >= total:
                print()  # New line when complete
        
        return callback


if __name__ == "__main__":
    # Example usage
    extractor = ArchiveExtractor()
    
    # Test archive extraction
    # result = extractor.extract_archive(
    #     "sample.zip",
    #     "extracted/",
    #     progress_callback=create_progress_callback()
    # )
    
    print(f"Supported formats: {extractor.supported_formats}")
    print("Archive extractor utilities ready!")