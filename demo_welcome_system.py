"""
Demo script showing the welcome screen and recommendation system
This script demonstrates the key features without launching the full Vbot application
"""

import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from utils.welcome_screen import AvatarRecommender
from utils.user_preferences import UserPreferences

def demo_recommendation_system():
    """Demonstrate the recommendation system"""
    print("🤖 VBOT AVATAR RECOMMENDATION SYSTEM DEMO")
    print("=" * 50)
    
    recommender = AvatarRecommender()
    
    # Show available avatars
    print("\n📋 Available Avatars:")
    avatars = recommender.get_all_avatars()
    for i, avatar in enumerate(avatars, 1):
        info = recommender.get_avatar_info(avatar)
        print(f"{i}. {info['name']} - {info['personality']}")
    
    print("\n" + "=" * 50)
    print("🎯 RECOMMENDATION SCENARIOS")
    print("=" * 50)
    
    # Test different user scenarios
    scenarios = [
        {
            "name": "Adventure Seeker",
            "preferences": {"gender": "female", "personality": ["energetic", "curious"]},
            "description": "Loves mystery and exploration"
        },
        {
            "name": "Intellectual User",
            "preferences": {"gender": "male", "personality": ["calm", "intellectual"]},
            "description": "Enjoys deep, thoughtful conversations"
        },
        {
            "name": "Casual User",
            "preferences": {"gender": "", "personality": ["playful", "friendly"]},
            "description": "Wants fun, light-hearted interactions"
        },
        {
            "name": "Creative Person",
            "preferences": {"gender": "female", "personality": ["mysterious", "artistic"]},
            "description": "Interested in art and creativity"
        },
        {
            "name": "Support Seeker",
            "preferences": {"gender": "male", "personality": ["reliable", "supportive"]},
            "description": "Needs guidance and steady support"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n👤 User Profile: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Preferences: {scenario['preferences']}")
        
        recommendation = recommender.recommend_avatar(scenario['preferences'])
        avatar_info = recommender.get_avatar_info(recommendation)
        
        print(f"   ✨ Recommended: {avatar_info['name']}")
        print(f"   📝 Reason: {avatar_info['description']}")

def demo_user_preferences():
    """Demonstrate the user preferences system"""
    print("\n" + "=" * 50)
    print("💾 USER PREFERENCES SYSTEM DEMO")
    print("=" * 50)
    
    # Create a test preferences instance
    prefs = UserPreferences()
    
    print(f"\n📊 Current Status:")
    print(f"   First time user: {prefs.get_preference('first_time_user', True)}")
    print(f"   Show welcome screen: {prefs.should_show_welcome_screen()}")
    print(f"   Last selected avatar: {prefs.get_last_selected_avatar() or 'None'}")
    print(f"   Total sessions: {prefs.get_preference('statistics.total_sessions', 0)}")
    
    # Simulate some user interactions
    print(f"\n🎭 Simulating User Interactions...")
    
    # User selects different avatars over time
    selections = ["Amelia", "Gura", "Amelia", "Shiori", "Amelia"]
    
    for i, avatar in enumerate(selections, 1):
        prefs.record_avatar_selection(avatar)
        print(f"   Session {i}: Selected {avatar}")
    
    # Show updated statistics
    print(f"\n📈 Updated Statistics:")
    print(f"   Total sessions: {prefs.get_preference('statistics.total_sessions', 0)}")
    print(f"   Last selected: {prefs.get_last_selected_avatar()}")
    print(f"   Favorite avatar: {prefs.get_preference('statistics.favorite_avatar')}")
    print(f"   First time user: {prefs.get_preference('first_time_user', True)}")
    
    # Show selection history
    history = prefs.get_preference('avatar_selection_history', [])
    print(f"\n📚 Selection History (last {len(history)} selections):")
    for record in history[-5:]:  # Show last 5
        print(f"   {record['avatar']} - {record['timestamp'][:19]}")

def demo_enhanced_recommendations():
    """Demonstrate enhanced recommendations with user history"""
    print("\n" + "=" * 50)
    print("🧠 ENHANCED RECOMMENDATIONS WITH HISTORY")
    print("=" * 50)
    
    recommender = AvatarRecommender()
    
    # Create a user with history
    user_profile = {
        "gender": "female",
        "personality": ["energetic", "curious"],
        "favorite_avatar": "Amelia",
        "selection_history": [
            {"avatar": "Amelia"},
            {"avatar": "Amelia"},
            {"avatar": "Gura"},
            {"avatar": "Amelia"}  # Recent selections
        ]
    }
    
    print(f"👤 User Profile:")
    print(f"   Preferred gender: {user_profile['gender']}")
    print(f"   Personality traits: {user_profile['personality']}")
    print(f"   Favorite avatar: {user_profile['favorite_avatar']}")
    print(f"   Recent selections: {[r['avatar'] for r in user_profile['selection_history'][-3:]]}")
    
    recommendation = recommender.recommend_avatar(user_profile)
    
    print(f"\n✨ Recommendation: {recommendation}")
    
    # Explain the recommendation
    if recommendation == user_profile['favorite_avatar']:
        print(f"   📝 Reason: Matches favorite avatar and personality preferences")
    else:
        print(f"   📝 Reason: Encouraging variety while matching personality traits")
    
    avatar_info = recommender.get_avatar_info(recommendation)
    print(f"   📋 Avatar info: {avatar_info['description']}")

def main():
    """Run the complete demo"""
    print("🚀 Welcome to the Vbot Avatar System Demo!")
    print("This demo showcases the new welcome screen and recommendation features.\n")
    
    try:
        # Demo the recommendation system
        demo_recommendation_system()
        
        # Demo user preferences
        demo_user_preferences()
        
        # Demo enhanced recommendations
        demo_enhanced_recommendations()
        
        print("\n" + "=" * 50)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("\nKey Features Demonstrated:")
        print("• Avatar recommendation based on user preferences")
        print("• User preference tracking and persistence")
        print("• Smart recommendations using usage history")
        print("• Variety encouragement to explore different avatars")
        print("\nTo see the full welcome screen UI, run: python Vbot.py")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
