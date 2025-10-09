"""
Resource path management for bundled and unbundled execution.
Handles PyInstaller frozen executable detection and path resolution.
"""
import os
import sys
from pathlib import Path


def get_base_path():
    """
    Get the base path for resources.
    
    Returns:
        Path: Base directory path (handles both frozen and unfrozen execution)
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running as PyInstaller bundle
        base = Path(sys._MEIPASS)
    else:
        # Running as normal Python script
        base = Path(__file__).parent.parent
    
    return base


def get_asset_path(*paths):
    """
    Get path to asset file/directory.
    
    Args:
        *paths: Path components to join
        
    Returns:
        Path: Full path to asset
        
    Example:
        get_asset_path('model', 'Amelia', 'character_model.yaml')
    """
    base = get_base_path()
    return base / 'asset' / Path(*paths)


def get_styletts2_path():
    """Get path to StyleTTS2 directory."""
    base = get_base_path()
    return base / 'StyleTTS2'


def get_tha4_path():
    """Get path to THA4 directory."""
    base = get_base_path()
    return base / 'tha4'


def get_utils_path():
    """Get path to utils directory."""
    base = get_base_path()
    return base / 'utils'


def get_cache_path(*paths):
    """
    Get path to cache directory (should be writable).
    
    Args:
        *paths: Path components to join
        
    Returns:
        Path: Full path to cache location
    """
    if getattr(sys, 'frozen', False):
        # For bundled exe, use user's AppData folder
        cache_base = Path(os.getenv('LOCALAPPDATA', os.path.expanduser('~'))) / 'Vbot' / 'cache'
    else:
        # For development, use project cache
        cache_base = get_base_path() / 'cache'
    
    cache_base.mkdir(parents=True, exist_ok=True)
    
    if paths:
        full_path = cache_base / Path(*paths)
        full_path.mkdir(parents=True, exist_ok=True)
        return full_path
    
    return cache_base


def get_output_path(*paths):
    """
    Get path to output directory (should be writable).
    
    Args:
        *paths: Path components to join
        
    Returns:
        Path: Full path to output location
    """
    if getattr(sys, 'frozen', False):
        # For bundled exe, use user's Documents folder
        output_base = Path(os.path.expanduser('~')) / 'Documents' / 'Vbot' / 'outputs'
    else:
        # For development, use project outputs
        output_base = get_asset_path('outputs')
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    if paths:
        full_path = output_base / Path(*paths)
        full_path.mkdir(parents=True, exist_ok=True)
        return full_path
    
    return output_base


def is_frozen():
    """Check if running as frozen executable."""
    return getattr(sys, 'frozen', False)


def get_executable_dir():
    """Get directory containing the executable or script."""
    if is_frozen():
        return Path(sys.executable).parent
    else:
        return get_base_path()


# Convenience functions for common paths
def get_model_path(model_name, *paths):
    """Get path to character model files."""
    return get_asset_path('model', model_name, *paths)


def get_ref_sound_path(model_name, *paths):
    """Get path to reference sound files."""
    return get_asset_path('ref_sound', model_name, *paths)


def get_background_path(model_name, *paths):
    """Get path to background images."""
    return get_asset_path('Background', model_name, *paths)


# Add to sys.path if needed
def setup_python_path():
    """Add necessary directories to Python path."""
    base = get_base_path()
    
    # Add base directory
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    
    # Add StyleTTS2 directory
    styletts2 = get_styletts2_path()
    if str(styletts2) not in sys.path:
        sys.path.insert(0, str(styletts2))
    
    # Add tha4 directory
    tha4 = get_tha4_path()
    if str(tha4) not in sys.path:
        sys.path.insert(0, str(tha4))


# Initialize paths on import
setup_python_path()
