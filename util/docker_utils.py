import docker
import requests
import time


class DockerHandler:
    def __init__(self):
        self.client = docker.from_env()
        self.setup_time = self.ensure_ollama_container()
        print(f"Docker setup took {self.setup_time:.2f}s")

    def ensure_ollama_container(self):
        """Ensure Ollama Docker container is running with GPU support"""
        setup_start = time.time()
        try:
            # Check if container exists and is running
            try:
                container = self.client.containers.get("ollama")
                if container.status != "running":
                    print("Starting existing Ollama container...")
                    container.start()
            except docker.errors.NotFound:
                print("Creating new Ollama container...")
                # Create and start the container with GPU support
                self.client.containers.run(
                    "ollama/ollama",
                    name="ollama",
                    detach=True,
                    runtime="nvidia",
                    environment=["NVIDIA_VISIBLE_DEVICES=all"],
                    volumes={"ollama": {"bind": "/root/.ollama", "mode": "rw"}},
                    ports={"11434/tcp": 11434},
                )

            # Wait for Ollama API to be ready
            self._wait_for_ollama()

            # Check if Stheno model exists
            container = self.client.containers.get("ollama")
            result = container.exec_run("ollama list")
            if "stheno" not in result.output.decode():
                print("Stheno model not found. Setting up...")
                # Pull the model from Hugging Face
                print("Pulling model from Hugging Face...")
                container.exec_run(
                    "ollama pull hf.co/featherless-ai-quants/bluuwhale-L3-SthenoMaidBlackroot-8B-V1-GGUF:bluuwhale-L3-SthenoMaidBlackroot-8B-V1-Q4_K_M.gguf"
                )

                # Copy to Stheno
                print("Setting up as Stheno...")
                container.exec_run(
                    "ollama cp hf.co/featherless-ai-quants/bluuwhale-L3-SthenoMaidBlackroot-8B-V1-GGUF:bluuwhale-L3-SthenoMaidBlackroot-8B-V1-Q4_K_M.gguf stheno"
                )

                # Clean up
                container.exec_run(
                    'ollama rm "hf.co/featherless-ai-quants/bluuwhale-L3-SthenoMaidBlackroot-8B-V1-GGUF:bluuwhale-L3-SthenoMaidBlackroot-8B-V1-Q4_K_M.gguf"'
                )
                print("Stheno model setup complete")
            else:
                print("Stheno model already exists")

            print("Ollama container is ready with GPU support")
            return time.time() - setup_start

        except Exception as e:
            print(f"Error setting up Docker container: {str(e)}")
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
            container = self.client.containers.get("ollama")
            container.stop()
        except:
            pass
