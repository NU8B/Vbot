import time
from pathlib import Path
import sys
import warnings

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from utils.ollama_utils import OllamaHandler, SYSTEM_PROMPT, MAX_HISTORY

# Suppress all warnings
warnings.filterwarnings("ignore")


def test_ollama_chat():
    """Test basic Ollama chat functionality"""
    print("Starting Ollama chat test...")

    # Test initialization/warmup
    print("\nTesting warmup...")
    warmup_time = OllamaHandler.initialize()
    print(f"Warmup completed in {warmup_time:.2f} seconds")

    # Test basic chat functionality
    print("\nTesting chat functionality...")
    message_history = []
    test_prompts = [
        "Hello! Who are you?",
        "What do you like to do?",
        "Tell me about your detective work.",
    ]

    for prompt in test_prompts:
        print(f"\nUser: {prompt}")
        start_time = time.time()

        # Use the static method to test Ollama directly
        response = OllamaHandler.call_ollama_static(
            prompt=prompt,
            message_history=message_history,
            max_history=MAX_HISTORY,
            system_prompt=SYSTEM_PROMPT,
        )

        elapsed_time = time.time() - start_time
        print(f"Assistant: {response}")
        print(f"Response time: {elapsed_time:.2f} seconds")

        # Update message history if we got a response
        if response:
            message_history.append({"role": "user", "content": prompt})
            message_history.append({"role": "assistant", "content": response})
            print(f"Conversation history length: {len(message_history)}")

    print("\nTest completed successfully!")


if __name__ == "__main__":
    try:
        test_ollama_chat()
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        # Print more detailed error information
        import traceback

        print("\nDetailed error:")
        print(traceback.format_exc())
