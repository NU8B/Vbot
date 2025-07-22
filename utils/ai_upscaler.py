import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import logging
import time
from typing import Optional, Tuple
import urllib.request
import hashlib

try:
    # Try to import Real-ESRGAN components
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    logging.warning("Real-ESRGAN not available. Install with: pip install realesrgan")

class AIUpscaler:
    """Real-time AI upscaling for avatar images using Real-ESRGAN and fallback methods."""
    
    def __init__(self, device: str = "auto", model_type: str = "anime"):
        """
        Initialize the AI upscaler.
        
        Args:
            device: "cuda", "cpu", or "auto"
            model_type: "anime" for anime-specific model, "general" for general purpose
        """
        self.device = self._setup_device(device)
        self.model_type = model_type
        self.upscaler = None
        self.fallback_enabled = True
        
        # Model paths and URLs
        self.models_dir = Path("cache/upscaler_models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_configs = {
            "anime": {
                "name": "RealESRGAN_x2plus_anime",
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x2plus.pth",
                "file": "RealESRGAN_x2plus_anime.pth",
                "scale": 2,
                "hash": "58c5d9e9d8b4f6b47d2b9c6f8e3a4d5f"  # Example hash
            },
            "general": {
                "name": "RealESRGAN_x2plus",
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x2plus.pth",
                "file": "RealESRGAN_x2plus.pth",
                "scale": 2,
                "hash": "78c5d9e9d8b4f6b47d2b9c6f8e3a4d5f"  # Example hash
            }
        }
        
        # Performance tracking
        self.stats = {
            "total_frames": 0,
            "total_time": 0.0,
            "avg_fps": 0.0,
            "last_process_time": 0.0
        }
        
        # Initialize the upscaler
        self._initialize_upscaler()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup and validate the compute device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logging.info(f"Auto-selected CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logging.info("CUDA not available, using CPU")
        
        torch_device = torch.device(device)
        
        # Test device availability
        try:
            test_tensor = torch.zeros(1, device=torch_device)
            del test_tensor
            return torch_device
        except Exception as e:
            logging.warning(f"Device {device} failed, falling back to CPU: {e}")
            return torch.device("cpu")
    
    def _download_model(self, config: dict) -> Path:
        """Download model weights if not available locally."""
        model_path = self.models_dir / config["file"]
        
        if model_path.exists():
            logging.info(f"Model already exists: {model_path}")
            return model_path
        
        logging.info(f"Downloading {config['name']} model...")
        try:
            urllib.request.urlretrieve(config["url"], str(model_path))
            logging.info(f"Model downloaded successfully: {model_path}")
            return model_path
        except Exception as e:
            logging.error(f"Failed to download model: {e}")
            raise RuntimeError(f"Could not download upscaling model: {e}")
    
    def _initialize_upscaler(self):
        """Initialize the Real-ESRGAN upscaler."""
        if not REALESRGAN_AVAILABLE:
            logging.warning("Real-ESRGAN not available, using fallback upscaling")
            return
        
        try:
            config = self.model_configs[self.model_type]
            model_path = self._download_model(config)
            
            # Initialize Real-ESRGAN model
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=config["scale"]
            )
            
            # Create the upscaler
            self.upscaler = RealESRGANer(
                scale=config["scale"],
                model_path=str(model_path),
                model=model,
                tile=256,  # Tile size for memory efficiency
                tile_pad=10,
                pre_pad=0,
                half=True if self.device.type == "cuda" else False,  # FP16 for speed
                device=self.device
            )
            
            logging.info(f"Real-ESRGAN {config['name']} initialized successfully on {self.device}")
            
        except Exception as e:
            logging.error(f"Failed to initialize Real-ESRGAN: {e}")
            logging.info("Will use fallback upscaling methods")
            self.upscaler = None
    
    def _fallback_upscale(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Fallback upscaling using OpenCV with enhanced interpolation."""
        height, width = target_size
        
        # Use LANCZOS for better quality than bilinear
        upscaled = cv2.resize(
            image, 
            (width, height), 
            interpolation=cv2.INTER_LANCZOS4
        )
        
        # Apply sharpening filter for better perceived quality
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(upscaled, -1, kernel * 0.1)
        
        # Blend original and sharpened (subtle effect)
        result = cv2.addWeighted(upscaled, 0.8, sharpened, 0.2, 0)
        
        return result
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for upscaling."""
        # Ensure image is in the right format (H, W, C) with values [0, 255]
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Handle alpha channel for RGBA images
        if image.shape[2] == 4:
            # Separate RGB and alpha
            rgb = image[:, :, :3]
            alpha = image[:, :, 3]
            return rgb, alpha
        else:
            return image, None
    
    def _postprocess_image(self, upscaled_rgb: np.ndarray, alpha: Optional[np.ndarray], 
                          target_size: Tuple[int, int]) -> np.ndarray:
        """Postprocess upscaled image and recombine with alpha."""
        if alpha is not None:
            # Upscale alpha channel using the same method as RGB
            alpha_upscaled = cv2.resize(
                alpha, 
                (target_size[1], target_size[0]), 
                interpolation=cv2.INTER_LANCZOS4
            )
            
            # Recombine RGBA
            result = np.dstack([upscaled_rgb, alpha_upscaled])
        else:
            result = upscaled_rgb
        
        # Ensure output is uint8
        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def upscale(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Upscale an image to the target size.
        
        Args:
            image: Input image as numpy array (H, W, C) or (H, W, 4) for RGBA
            target_size: Target size as (height, width)
            
        Returns:
            Upscaled image as numpy array
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            rgb_image, alpha = self._preprocess_image(image)
            
            # Try Real-ESRGAN first
            if self.upscaler is not None:
                try:
                    # Real-ESRGAN expects BGR format
                    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                    
                    # Perform upscaling
                    upscaled_bgr, _ = self.upscaler.enhance(bgr_image, outscale=None)
                    
                    # Convert back to RGB
                    upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
                    
                    # Resize to exact target size if needed
                    if upscaled_rgb.shape[:2] != target_size:
                        upscaled_rgb = cv2.resize(
                            upscaled_rgb, 
                            (target_size[1], target_size[0]), 
                            interpolation=cv2.INTER_LANCZOS4
                        )
                    
                except Exception as e:
                    logging.warning(f"Real-ESRGAN failed, using fallback: {e}")
                    upscaled_rgb = self._fallback_upscale(rgb_image, target_size)
            else:
                # Use fallback method
                upscaled_rgb = self._fallback_upscale(rgb_image, target_size)
            
            # Postprocess and recombine with alpha
            result = self._postprocess_image(upscaled_rgb, alpha, target_size)
            
            # Update performance stats
            process_time = time.time() - start_time
            self._update_stats(process_time)
            
            return result
            
        except Exception as e:
            logging.error(f"Upscaling failed: {e}")
            # Emergency fallback: simple resize
            return cv2.resize(image, (target_size[1], target_size[0]))
    
    def _update_stats(self, process_time: float):
        """Update performance statistics."""
        self.stats["total_frames"] += 1
        self.stats["total_time"] += process_time
        self.stats["last_process_time"] = process_time
        self.stats["avg_fps"] = self.stats["total_frames"] / self.stats["total_time"]
    
    def get_performance_stats(self) -> dict:
        """Get current performance statistics."""
        return self.stats.copy()
    
    def is_available(self) -> bool:
        """Check if AI upscaling is available."""
        return self.upscaler is not None
    
    def get_info(self) -> dict:
        """Get upscaler information."""
        return {
            "device": str(self.device),
            "model_type": self.model_type,
            "realesrgan_available": REALESRGAN_AVAILABLE,
            "upscaler_initialized": self.upscaler is not None,
            "stats": self.get_performance_stats()
        } 