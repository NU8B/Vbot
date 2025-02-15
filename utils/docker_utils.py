import docker
import requests
import time
import os


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
                else:
                    print("Ollama container is already running")
            except docker.errors.NotFound:
                print("Creating new Ollama container...")
                # Create and start the container with GPU support and proper networking
                container = self.client.containers.run(
                    "ollama/ollama",
                    name="ollama",
                    detach=True,
                    # runtime="nvidia",  # Comment out GPU support
                    # environment=["NVIDIA_VISIBLE_DEVICES=all"],  # Comment out GPU environment
                    volumes={"ollama": {"bind": "/root/.ollama", "mode": "rw"}},
                    ports={"11434/tcp": 11434},
                    network_mode="bridge",  # Changed to bridge mode for better Docker networking
                )

            # Wait for container to be fully running
            container = self.client.containers.get("ollama")
            while container.status != "running":
                print(
                    f"Waiting for container to start... Current status: {container.status}"
                )
                time.sleep(1)
                container.reload()

            # Wait for Ollama API to be ready with better error handling
            self._wait_for_ollama()

            # Check if model exists and is loaded
            print("Checking model status...")
            container = self.client.containers.get("ollama")
            result = container.exec_run("ollama list")
            if "stheno" not in result.output.decode():
                print("Stheno model not found. Pulling model...")
                pull_result = container.exec_run("ollama pull stheno")
                if pull_result.exit_code != 0:
                    print(f"Error pulling model: {pull_result.output.decode()}")
                    raise Exception("Failed to pull Stheno model")
                print("Stheno model pulled successfully")

            return time.time() - setup_start

        except docker.errors.APIError as e:
            print(f"Docker API error: {str(e)}")
            raise
        except Exception as e:
            print(f"Error setting up Docker container: {str(e)}")
            raise

    def _wait_for_ollama(self, timeout=60, interval=1):
        """Wait for Ollama API to be ready with better error handling"""
        print("Waiting for Ollama API to be ready...")
        start_time = time.time()
        last_error = None

        while time.time() - start_time < timeout:
            try:
                # Try the root endpoint first (this should always work if server is up)
                response = requests.get("http://localhost:11434")
                if response.status_code == 200:
                    print("Ollama API is ready")
                    return

                # If root fails, try a model list request as fallback
                response = requests.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    print("Ollama API is ready (via tags endpoint)")
                    return

                last_error = f"Unexpected status code: {response.status_code}"
            except requests.ConnectionError:
                last_error = "Connection refused - container might still be starting"
            except Exception as e:
                last_error = str(e)

            print(
                f"Waiting for Ollama API... ({int(time.time() - start_time)}s) - {last_error}"
            )

            if time.time() - start_time + interval >= timeout:
                break

            time.sleep(interval)

        raise TimeoutError(f"Ollama API failed to become ready: {last_error}")

    def cleanup(self):
        """Cleanup method to stop the container"""
        try:
            container = self.client.containers.get("ollama")
            print("Stopping Ollama container...")
            container.stop()
            print("Ollama container stopped")
        except docker.errors.NotFound:
            print("Ollama container not found")
        except Exception as e:
            print(f"Error stopping container: {str(e)}")
