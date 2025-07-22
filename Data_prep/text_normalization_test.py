#!/usr/bin/env python3
"""
Text Normalization Test Script
Test the text cleaning and normalization functions from data_StyleTTS2.py
"""

import sys
import os
from pathlib import Path

# Add current directory to path to import from data_StyleTTS2
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_StyleTTS2 import normalize_text_for_tts, phonemize_text
except ImportError as e:
    print(f"Error importing from data_StyleTTS2.py: {e}")
    print("Make sure data_StyleTTS2.py is in the same directory")
    sys.exit(1)


def test_text_normalization(text):
    """Test the text normalization pipeline"""
    print(f"\n{'='*60}")
    print(f"ORIGINAL TEXT:")
    print(f"'{text}'")
    print(f"\n{'='*60}")

    # Step 1: Apply text normalization
    normalized = normalize_text_for_tts(text)
    print(f"NORMALIZED TEXT:")
    print(f"'{normalized}'")

    # Step 2: Show phonemization (if available)
    print(f"\n{'-'*60}")
    try:
        phonemes = phonemize_text(text)  # This calls normalize_text_for_tts internally
        if phonemes:
            print(f"PHONEMIZED TEXT:")
            print(f"'{phonemes}'")
        else:
            print("PHONEMIZATION FAILED")
    except Exception as e:
        print(f"PHONEMIZATION ERROR: {e}")

    print(f"{'='*60}\n")
    return normalized


def run_predefined_tests():
    """Run a set of predefined test cases"""
    test_cases = [
        "Hello world! How are you today?",
        "I have $42.50 in my wallet.",
        "The meeting is at 2:30 PM on 2024-01-15.",
        "Call me at 555-123-4567 or visit www.example.com.",
        "I scored 95% on my test!!!",
        "The temperature is 72°F outside.",
        "He's running at 15.5 mph.",
        "Visit 123 Main St. for the party.",
        "Dr. Smith will see you at 3:00.",
        "Mmmm. Hmmmm... That's interesting.",
        "I said uh... what was that?",
        "The file is 2.5 GB in size.",
        "It happened in the year 2004.",
        "Chapter IV discusses the results.",
        "I can't believe it's already 10:30 AM!",
        "The price increased by 15.7%.",
        "Send an email to test@example.com.",
        "He ran the 100m in 9.58 seconds.",
        "The address is 1600 Pennsylvania Ave.",
        "Room 21B is on the 3rd floor.",
    ]

    print("Running predefined test cases...")
    print("=" * 80)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTEST CASE {i}:")
        test_text_normalization(test_case)


def interactive_mode():
    """Interactive mode for testing custom text"""
    print("\n" + "=" * 80)
    print("INTERACTIVE TEXT NORMALIZATION TESTING")
    print("=" * 80)
    print("Enter text to see how it will be normalized for TTS training.")
    print("Type 'quit', 'exit', or press Ctrl+C to exit.")
    print("Type 'tests' to run predefined test cases.")
    print("-" * 80)

    while True:
        try:
            user_input = input("\nEnter text to test: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            elif user_input.lower() == "tests":
                run_predefined_tests()
                continue
            elif not user_input:
                print("Please enter some text to test.")
                continue

            test_text_normalization(user_input)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function"""
    print("Text Normalization Tester for StyleTTS2")
    print("This script tests the text cleaning functions from data_StyleTTS2.py")

    if len(sys.argv) > 1:
        # If arguments provided, test those
        text_to_test = " ".join(sys.argv[1:])
        test_text_normalization(text_to_test)
    else:
        # Run in interactive mode
        try:
            interactive_mode()
        except Exception as e:
            print(f"Error in interactive mode: {e}")


if __name__ == "__main__":
    main()
