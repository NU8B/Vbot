# Setup Instructions for Windows 10

## Prerequisites

### 1. NVIDIA GPU Driver
1. Visit [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
2. Select your GPU model and Windows 10
3. Download and install the driver
4. Verify installation by running `nvidia-smi` in Command Prompt

### 2. Docker Desktop
1. Download [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. During installation:
   - Check "Use WSL 2 instead of Hyper-V"
   - Complete the installation
   - Restart your computer if prompted

## Setup Steps

### 1. Install WSL 2
1. Open PowerShell as Administrator
2. Run:
   ```powershell
   wsl --install -d Ubuntu
   ```
3. When prompted:
   - Create a username (use lowercase letters only, e.g., `user`)
   - Set a password (characters won't be visible as you type)
   - Wait for Ubuntu installation to complete

### 2. Configure Docker Desktop
1. Open Docker Desktop
2. Go to Settings (gear icon)
3. Under "Resources" → "WSL Integration":
   - Enable "Use the WSL 2 based engine"
   - Enable "Ubuntu" in the distribution list
   - Click "Apply"

4. Under "Docker Engine":
   - Click the edit button (it's a JSON file)
   - Replace **ALL** content with this exact configuration:
   ```json
   {
     "builder": {
       "gc": {
         "defaultKeepStorage": "20GB",
         "enabled": true
       }
     },
     "experimental": false,
     "default-runtime": "nvidia",
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     }
   }
   ```
   - Click "Apply & Restart"
   - Wait for Docker Desktop to restart

### 3. Install Python Requirements
1. Open Command Prompt
2. Run:
   ```cmd
   pip install docker nltk requests
   ```

## Running the Application


### First Run
- The first run will take longer as it needs to:
  1. Create the Docker container
  2. Pull the Stheno model (~5GB)
  3. Set up GPU acceleration
- Subsequent runs will be faster as the model is cached

## Troubleshooting

### Common Issues

1. **Docker Desktop WSL Error**
   - Open PowerShell as Administrator
   - Run: `wsl --update`
   - Restart Docker Desktop

2. **NVIDIA Driver Issues**
   - Verify GPU is detected: Run `nvidia-smi` in Command Prompt
   - Update to latest driver if needed

3. **Docker Container Not Starting**
   - Check Docker Desktop is running
   - Try restarting Docker Desktop
   - Ensure WSL 2 is properly installed

### Need Help?
If you encounter any issues:
1. Check the error message
2. Verify all prerequisites are installed
3. Try restarting Docker Desktop
4. Make sure your GPU drivers are up to date 