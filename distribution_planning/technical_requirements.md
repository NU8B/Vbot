# Vbot Technical Requirements Analysis

## System Requirements Matrix

### Minimum Requirements
| Component | Specification | Justification |
|-----------|---------------|---------------|
| **OS** | Windows 10 64-bit (Build 19041+) | WSL2 requirement for Docker |
| **CPU** | Intel i5-8400 / AMD Ryzen 5 2600 | Multi-threading for audio processing |
| **RAM** | 16GB DDR4 | 6-8GB for models + 4GB for OS + 4GB buffer |
| **GPU** | NVIDIA GTX 1070 (8GB VRAM) | CUDA compute capability 6.1+ |
| **Storage** | 50GB free space (SSD recommended) | Models + cache + temp files |
| **Network** | Broadband (25+ Mbps) | Initial model downloads |

### Recommended Requirements  
| Component | Specification | Benefits |
|-----------|---------------|----------|
| **OS** | Windows 11 64-bit | Better WSL2 integration |
| **CPU** | Intel i7-10700K / AMD Ryzen 7 3700X | Faster audio processing |
| **RAM** | 32GB DDR4 | Better multitasking, larger model cache |
| **GPU** | NVIDIA RTX 3070 (8GB+) | Faster inference, better performance |
| **Storage** | 100GB NVMe SSD | Faster model loading |
| **Network** | Gigabit ethernet | Faster downloads |

## Software Dependencies Analysis

### Critical System Dependencies
```yaml
Required for Core Functionality:
  - Windows 10/11 (64-bit)
  - NVIDIA GPU Driver (545.84+)
  - CUDA Toolkit 12.1
  - cuDNN 8.9.7 (exact version)
  - Docker Desktop 4.0+
  - WSL 2 with Ubuntu
  - Microsoft C++ Build Tools 2019+
  - espeak (phonemizer dependency)

Optional but Recommended:
  - Git for Windows (development)
  - 7-Zip (archive extraction)
  - Windows Terminal (better CLI)
```

### Python Environment Requirements
```yaml
Python Version: 3.10.x (exact)
  Reason: Compatibility with PyTorch 2.5.1 and transformers

Key Package Constraints:
  - torch==2.5.1+cu121 (CUDA 12.1 specific)
  - transformers==4.46.3 (API compatibility)
  - wxPython==4.2.2 (GUI framework)
  - docker==7.1.0 (container management)
  
Total Packages: 111 (see requirements.txt)
Install Size: ~8GB (including PyTorch)
```

### Model Requirements
```yaml
StyleTTS2 Models (per character):
  Size: 1-2GB each
  Source: HuggingFace (nonoJDWAOIDAWKDA repos)
  Format: PyTorch .pth files
  Dependencies: ASR, F0, PLBERT utilities (~500MB shared)

Ollama LLM Model:
  Model: Stheno (bluuwhale-L3-SthenoMaidBlackroot-8B)
  Size: ~5GB
  Format: GGUF (quantized)
  Runtime: Docker container

THA4 Character Models:
  Size: ~100MB per character
  Format: PyTorch .pth + YAML config
  Location: asset/model/{character}/
```

## Hardware Compatibility Matrix

### NVIDIA GPU Compatibility
| GPU Series | VRAM | CUDA Compute | Status | Notes |
|------------|------|--------------|--------|-------|
| **RTX 40 Series** | 8GB+ | 8.9 | ✅ Excellent | Best performance |
| **RTX 30 Series** | 8GB+ | 8.6 | ✅ Excellent | Recommended |
| **RTX 20 Series** | 8GB+ | 7.5 | ✅ Good | Solid performance |
| **GTX 16 Series** | 6GB+ | 7.5 | ⚠️ Limited | Minimum viable |
| **GTX 10 Series** | 8GB+ | 6.1 | ⚠️ Limited | GTX 1070+ only |
| **Older GPUs** | Any | <6.1 | ❌ Incompatible | Not supported |
| **AMD GPUs** | Any | N/A | ❌ Incompatible | CUDA required |
| **Intel GPUs** | Any | N/A | ❌ Incompatible | CUDA required |

### CPU Compatibility
```yaml
Intel:
  Minimum: 8th gen (Coffee Lake)
  Recommended: 10th gen+ (Comet Lake+)
  
AMD:
  Minimum: Ryzen 2000 series (Zen+)
  Recommended: Ryzen 3000+ (Zen 2+)
  
Architecture: x64 only (no 32-bit support)
Cores: 4+ physical cores recommended
Threads: 8+ threads for optimal performance
```

### Memory Requirements Detail
```yaml
RAM Breakdown:
  - Windows OS: 4GB baseline
  - Python Runtime: 1GB
  - PyTorch + CUDA: 2GB
  - StyleTTS2 Models: 2-3GB (loaded)
  - Ollama Container: 6GB (Stheno model)
  - Audio Buffers: 500MB
  - GUI Framework: 500MB
  - Buffer/Cache: 2GB
  
Total: 16-18GB typical usage
Peak: 20GB+ during model switching
```

### Storage Requirements Detail
```yaml
Installation Footprint:
  - Python Environment: 8GB
  - Vbot Application: 2GB
  - Docker Images: 3GB
  - Models (5 characters): 10GB
  - Cache Directory: 5GB
  - Temp Files: 2GB
  
Total: ~30GB minimum
Recommended: 50GB+ for updates and additional models
```

## Network Requirements

### Bandwidth Requirements
```yaml
Initial Setup:
  - Python packages: 2GB download
  - Docker images: 1GB download  
  - Models: 6-7GB download
  - Total: ~10GB first run
  
Runtime:
  - Ollama API: Minimal (local)
  - Model updates: Occasional (1-2GB)
  - Telemetry: <1MB/day
  
Recommended: 25+ Mbps for reasonable setup time
Minimum: 10 Mbps (slower initial setup)
```

### Firewall/Port Requirements
```yaml
Outbound (Required):
  - HTTPS (443): HuggingFace, Docker Hub, GitHub
  - HTTP (80): Package repositories
  
Local (Docker):
  - 11434: Ollama API (localhost only)
  - Dynamic: Docker internal networking
  
No inbound ports required (desktop application)
```

## Platform-Specific Considerations

### Windows 10 vs Windows 11
```yaml
Windows 10:
  - Requires manual WSL2 setup
  - Docker Desktop compatibility varies
  - Some GPU driver limitations
  - Supported but more setup steps
  
Windows 11:
  - Better WSL2 integration
  - Improved Docker Desktop support
  - Latest GPU driver support
  - Preferred platform
```

### WSL2 Requirements
```yaml
Prerequisites:
  - Windows 10 Build 19041+ or Windows 11
  - Virtualization enabled in BIOS
  - Hyper-V compatible CPU
  - 4GB+ RAM available for WSL
  
Installation:
  - wsl --install -d Ubuntu
  - Docker Desktop WSL2 backend
  - GPU passthrough for CUDA
```

### Docker Desktop Considerations
```yaml
System Requirements:
  - WSL2 or Hyper-V backend
  - 4GB+ RAM for containers
  - Virtualization support
  
Configuration:
  - NVIDIA Container Runtime
  - GPU device passthrough
  - Volume mounts for model persistence
  - Network bridge configuration
```

## Performance Characteristics

### Startup Performance
```yaml
Cold Start (First Run):
  - Docker container creation: 30-60s
  - Model downloads: 10-30 minutes (network dependent)
  - Model loading: 30-60s
  - Total: 15-45 minutes
  
Warm Start (Cached):
  - Docker startup: 10-20s
  - Model loading: 15-30s
  - GUI initialization: 5-10s
  - Total: 30-60s
  
Hot Start (Already running):
  - Character switch: 10-15s
  - New conversation: <1s
```

### Runtime Performance
```yaml
Response Generation:
  - Text processing: 100-500ms
  - LLM inference: 1-3s
  - TTS synthesis: 2-4s
  - Avatar animation: Real-time (30 FPS)
  - Total latency: 3-8s end-to-end
  
Resource Usage:
  - CPU: 20-40% (during inference)
  - GPU: 60-80% VRAM, 30-50% compute
  - RAM: 16-20GB steady state
  - Disk I/O: Minimal after startup
```

### Scalability Limits
```yaml
Concurrent Users: 1 (desktop application)
Model Switching: 5 characters supported
Conversation Length: Unlimited (with history pruning)
Audio Quality: 22kHz, 16-bit (configurable)
Animation Quality: 512x512 @ 30 FPS
```

## Compatibility Testing Matrix

### Test Configurations
```yaml
Primary Test Systems:
  1. Windows 11 + RTX 3070 + 32GB RAM (optimal)
  2. Windows 10 + GTX 1070 + 16GB RAM (minimum)
  3. Windows 11 + RTX 4090 + 64GB RAM (high-end)
  
Edge Case Systems:
  1. Windows 10 + GTX 1060 6GB (below minimum)
  2. Windows 11 + No NVIDIA GPU (incompatible)
  3. Laptop with Optimus (driver complexity)
  
Virtual Environments:
  1. VMware with GPU passthrough
  2. Hyper-V with discrete assignment
  3. Cloud instances (AWS, Azure)
```

### Known Compatibility Issues
```yaml
Hardware Issues:
  - GTX 1060 6GB: Insufficient VRAM
  - Laptop Optimus: Driver switching problems
  - Multiple GPUs: CUDA device selection
  
Software Issues:
  - Antivirus interference with PyTorch
  - Windows Defender blocking Docker
  - Outdated NVIDIA drivers
  - cuDNN version mismatches
  
Network Issues:
  - Corporate firewalls blocking downloads
  - Proxy servers interfering with Docker
  - DNS resolution for HuggingFace
```

## Distribution Impact Analysis

### Installer Size Implications
```yaml
PyInstaller Bundle:
  - Python runtime: 100MB
  - Core dependencies: 400MB
  - GPU libraries: 300MB
  - Application code: 50MB
  - Assets (minimal): 50MB
  - Total: ~900MB executable
  
Separate Downloads:
  - Models: 6-7GB (first run)
  - Docker images: 1GB (auto-pull)
  - Updates: 100-500MB (periodic)
```

### Installation Time Estimates
```yaml
Fast System (SSD + Gigabit):
  - Installer download: 2-5 minutes
  - Installation: 2-3 minutes
  - Model downloads: 5-10 minutes
  - Total: 10-20 minutes
  
Slow System (HDD + 25 Mbps):
  - Installer download: 5-10 minutes
  - Installation: 5-10 minutes
  - Model downloads: 20-40 minutes
  - Total: 30-60 minutes
```

---

**This analysis informs the distribution strategy and helps set realistic user expectations.**
