"""
Test script for the avatar recommender system only (no UI)
"""

import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from utils.welcome_screen import AvatarRecommender

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

if __name__ == "__main__":
    print("Avatar Recommender Test")
    print("=" * 40)
    
    # Test the recommender system
    test_avatar_recommender()
    print("\nRecommender test completed successfully!")
