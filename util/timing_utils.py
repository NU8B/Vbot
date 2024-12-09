import threading
import time


class ParallelInitializer:
    def __init__(self):
        self.results = {}
        self.threads = []
        self.start_time = time.time()

    def add_task(self, name, func, *args, **kwargs):
        """Add a task to be executed in parallel"""

        def wrapper():
            start = time.time()
            result = func(*args, **kwargs)
            self.results[name] = {"result": result, "time": time.time() - start}

        thread = threading.Thread(target=wrapper, daemon=True)
        self.threads.append(thread)
        return self

    def run(self):
        """Run all tasks in parallel and wait for completion"""
        # Start all threads
        for thread in self.threads:
            thread.start()

        # Wait for all threads to complete
        for thread in self.threads:
            thread.join()

        self.total_time = time.time() - self.start_time
        return self.results

    def get_timing(self, name):
        """Get the execution time for a specific task"""
        return self.results.get(name, {}).get("time", 0)

    def get_result(self, name):
        """Get the result for a specific task"""
        return self.results.get(name, {}).get("result", None)
