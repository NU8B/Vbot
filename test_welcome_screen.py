"""
Test script for the welcome screen functionality
"""

import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from utils.welcome_screen import show_welcome_screen, AvatarRecommender

def test_avatar_recommender():
    """Test the avatar recommendation system"""
    print("Testing Avatar Recommender...")
    
    recommender = AvatarRecommender()
    
    # Test getting all avatars
    avatars = recommender.get_all_avatars()
    print(f"Available avatars: {avatars}")
    
    # Test getting avatar info
    for avatar in avatars:
        info = recommender.get_avatar_info(avatar)
        print(f"\n{avatar}:")
        print(f"  Name: {info.get('name', 'N/A')}")
        print(f"  Personality: {info.get('personality', 'N/A')}")
        print(f"  Voice: {info.get('voice_type', 'N/A')}")
        print(f"  Description: {info.get('description', 'N/A')}")
    
    # Test recommendations
    print("\n" + "="*50)
    print("Testing Recommendations:")
    
    test_preferences = [
        {"gender": "female", "personality": ["energetic", "curious"]},
        {"gender": "male", "personality": ["calm", "intellectual"]},
        {"gender": "", "personality": ["playful", "friendly"]},
        {"gender": "female", "personality": ["mysterious", "artistic"]},
    ]
    
    for i, prefs in enumerate(test_preferences, 1):
        recommendation = recommender.recommend_avatar(prefs)
        print(f"\nTest {i}: {prefs}")
        print(f"Recommendation: {recommendation}")

def test_welcome_screen():
    """Test the welcome screen UI"""
    print("\nTesting Welcome Screen UI...")
    
    def on_avatar_selected(avatar_name):
        print(f"Selected avatar: {avatar_name}")
        print("Welcome screen test completed successfully!")
    
    try:
        show_welcome_screen(on_avatar_selected)
    except Exception as e:
        print(f"Error testing welcome screen: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Welcome Screen Test Suite")
    print("=" * 40)
    
    # Test the recommender system
    test_avatar_recommender()
    
    # Test the UI (comment out if running in headless environment)
    test_welcome_screen()
