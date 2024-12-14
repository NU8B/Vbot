import docker
import requests
import time
from pathlib import Path


class DockerHandler:
    def __init__(self):
        setup_start = time.time()
        self.client = docker.from_env()
        self.container = None

        try:
            self._ensure_ollama_container()
            self._ensure_model_exists()
            self.setup_time = time.time() - setup_start
            print(f"Docker setup took {self.setup_time:.2f}s")
        except Exception as e:
            print(f"Error during Docker setup: {str(e)}")
            raise

    def _ensure_ollama_container(self):
        """Ensure Ollama container is running"""
        try:
            # Try to get existing container
            containers = self.client.containers.list(
                filters={"name": "ollama", "status": "running"}
            )

            if containers:
                self.container = containers[0]
                return

            # Try to start existing but stopped container
            containers = self.client.containers.list(
                all=True, filters={"name": "ollama"}
            )
            if containers:
                self.container = containers[0]
                self.container.start()
                return

            # Create and start new container if none exists
            self.container = self.client.containers.run(
                "ollama/ollama",
                name="ollama",
                detach=True,
                ports={"11434/tcp": 11434},
                volumes={
                    str(Path.home() / ".ollama"): {
                        "bind": "/root/.ollama",
                        "mode": "rw",
                    }
                },
                device_requests=[
                    docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                ],
            )

        except Exception as e:
            print(f"Error setting up Ollama container: {str(e)}")
            raise

    def _ensure_model_exists(self):
        """Ensure Stheno model exists and pull if needed"""
        try:
            result = self.container.exec_run("ollama list")
            if "stheno" not in result.output.decode():
                # Pull and setup model
                self.container.exec_run(
                    "ollama pull hf.co/featherless-ai-quants/bluuwhale-L3-SthenoMaidBlackroot-8B-V1-GGUF:bluuwhale-L3-SthenoMaidBlackroot-8B-V1-Q4_K_M.gguf"
                )
                self.container.exec_run(
                    "ollama cp hf.co/featherless-ai-quants/bluuwhale-L3-SthenoMaidBlackroot-8B-V1-GGUF:bluuwhale-L3-SthenoMaidBlackroot-8B-V1-Q4_K_M.gguf stheno"
                )
                self.container.exec_run(
                    'ollama rm "hf.co/featherless-ai-quants/bluuwhale-L3-SthenoMaidBlackroot-8B-V1-GGUF:bluuwhale-L3-SthenoMaidBlackroot-8B-V1-Q4_K_M.gguf"'
                )
        except Exception as e:
            print(f"Error ensuring model exists: {str(e)}")
            raise

    def _wait_for_ollama(self, timeout=60):
        """Wait for Ollama API to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                requests.get("http://localhost:11434/api/health")
                print("Ollama API is ready")
                return
            except requests.exceptions.RequestException:
                time.sleep(1)
        raise TimeoutError("Ollama API failed to become ready")

    def cleanup(self):
        """Cleanup method to stop the container"""
        try:
            if self.container:
                self.container.stop()
        except:
            pass
