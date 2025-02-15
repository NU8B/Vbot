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
                container = self.client.containers.run(
                    "ollama/ollama",
                    name="ollama",
                    detach=True,
                    volumes={
                        "ollama": {"bind": "/root/.ollama", "mode": "rw"}
                    },
                    ports={"11434/tcp": 11434},
                    network_mode="bridge",  # Explicitly set network mode
                    environment={
                        "OLLAMA_HOST": "0.0.0.0",  # Listen on all interfaces
                        "OLLAMA_ORIGINS": "*"       # Allow all origins
                    }
                )

            # Wait for container to be fully ready
            time.sleep(5)
            
            # Wait for Ollama API
            self._wait_for_ollama()
            
            # Verify model is loaded
            print("Verifying Mistral model...")
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json()
                if not any(model.get('name', '').startswith('mistral') for model in models.get('models', [])):
                    print("Pulling Mistral model...")
                    pull_response = requests.post(
                        "http://localhost:11434/api/pull",
                        json={"name": "mistral"}
                    )
                    if pull_response.status_code != 200:
                        raise Exception("Failed to pull Mistral model")
            
            return time.time() - setup_start

        except Exception as e:
            print(f"Error setting up Docker container: {str(e)}")
            raise

    def _wait_for_ollama(self):
        """Wait for Ollama API to be ready with simplified check"""
        max_attempts = 30
        delay = 2
        
        print("Waiting for Ollama API to be ready...")
        for i in range(max_attempts):
            try:
                # Only test base endpoint first
                response = requests.get("http://localhost:11434/", timeout=5)
                if response.status_code == 200:
                    print("Ollama API is ready!")
                    return
            except requests.exceptions.RequestException as e:
                print(f"API test failed (attempt {i + 1}/{max_attempts}): {str(e)}")
            
            if i < max_attempts - 1:
                time.sleep(delay)
        
        raise Exception("Ollama API failed to become ready")

    def cleanup(self):
        """Cleanup method to stop the container"""
        try:
            container = self.client.containers.get("ollama")
            container.stop()
        except:
            pass
