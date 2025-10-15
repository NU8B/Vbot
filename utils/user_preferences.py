"""
User Preferences Management Module
Handles saving and loading user preferences for avatar selection and app settings
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime


class UserPreferences:
    """Manages user preferences and settings"""
    
    def __init__(self, config_dir: str = None):
        """
        Initialize user preferences manager
        
        Args:
            config_dir: Directory to store config files. If None, uses default.
        """
        if config_dir is None:
            # Use a config directory in the user's home folder
            self.config_dir = Path.home() / ".vbot"
        else:
            self.config_dir = Path(config_dir)
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        # Config file path
        self.config_file = self.config_dir / "user_preferences.json"
        
        # Default preferences
        self.default_preferences = {
            "last_selected_avatar": None,
            "show_welcome_screen": True,
            "first_time_user": True,
            "avatar_selection_history": [],
            "user_profile": {
                "preferred_gender": "",
                "preferred_personality_traits": [],
                "interaction_style": "casual"
            },
            "app_settings": {
                "window_size": "1280x720",
                "theme": "dark",
                "auto_start_voice": False,
                "show_subtitles": True
            },
            "statistics": {
                "total_sessions": 0,
                "favorite_avatar": None,
                "last_session_date": None
            }
        }
        
        # Load existing preferences or create new ones
        self.preferences = self.load_preferences()
    
    def load_preferences(self) -> Dict[str, Any]:
        """Load preferences from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_prefs = json.load(f)
                
                # Merge with defaults to ensure all keys exist
                preferences = self.default_preferences.copy()
                preferences.update(loaded_prefs)
                
                return preferences
            else:
                # First time user - create default config
                return self.default_preferences.copy()
        
        except Exception as e:
            print(f"Error loading preferences: {e}")
            return self.default_preferences.copy()
    
    def save_preferences(self) -> bool:
        """Save preferences to file"""
        try:
            # Update last session date
            self.preferences["statistics"]["last_session_date"] = datetime.now().isoformat()
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.preferences, f, indent=2, ensure_ascii=False)
            
            return True
        
        except Exception as e:
            print(f"Error saving preferences: {e}")
            return False
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a specific preference value"""
        keys = key.split('.')
        value = self.preferences
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_preference(self, key: str, value: Any) -> None:
        """Set a specific preference value"""
        keys = key.split('.')
        target = self.preferences
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        # Set the value
        target[keys[-1]] = value
    
    def should_show_welcome_screen(self) -> bool:
        """Determine if welcome screen should be shown"""
        # Show welcome screen if:
        # 1. User preference is set to show it, OR
        # 2. It's a first-time user, OR
        # 3. No avatar has been selected before
        return (
            self.get_preference("show_welcome_screen", True) or
            self.get_preference("first_time_user", True) or
            self.get_preference("last_selected_avatar") is None
        )
    
    def record_avatar_selection(self, avatar_name: str) -> None:
        """Record an avatar selection"""
        # Update last selected avatar
        self.set_preference("last_selected_avatar", avatar_name)
        
        # Add to selection history
        history = self.get_preference("avatar_selection_history", [])
        selection_record = {
            "avatar": avatar_name,
            "timestamp": datetime.now().isoformat()
        }
        history.append(selection_record)
        
        # Keep only last 10 selections
        if len(history) > 10:
            history = history[-10:]
        
        self.set_preference("avatar_selection_history", history)
        
        # Update statistics
        self.increment_session_count()
        self.update_favorite_avatar()
        
        # Mark as no longer first-time user
        self.set_preference("first_time_user", False)
    
    def increment_session_count(self) -> None:
        """Increment the total session count"""
        current_count = self.get_preference("statistics.total_sessions", 0)
        self.set_preference("statistics.total_sessions", current_count + 1)
    
    def update_favorite_avatar(self) -> None:
        """Update the favorite avatar based on usage history"""
        history = self.get_preference("avatar_selection_history", [])
        
        if not history:
            return
        
        # Count avatar usage
        avatar_counts = {}
        for record in history:
            avatar = record["avatar"]
            avatar_counts[avatar] = avatar_counts.get(avatar, 0) + 1
        
        # Find most used avatar
        if avatar_counts:
            favorite = max(avatar_counts.items(), key=lambda x: x[1])
            self.set_preference("statistics.favorite_avatar", favorite[0])
    
    def get_last_selected_avatar(self) -> Optional[str]:
        """Get the last selected avatar"""
        return self.get_preference("last_selected_avatar")
    
    def get_user_profile_for_recommendation(self) -> Dict[str, Any]:
        """Get user profile data for avatar recommendation"""
        return {
            "gender": self.get_preference("user_profile.preferred_gender", ""),
            "personality": self.get_preference("user_profile.preferred_personality_traits", []),
            "interaction_style": self.get_preference("user_profile.interaction_style", "casual"),
            "favorite_avatar": self.get_preference("statistics.favorite_avatar"),
            "selection_history": self.get_preference("avatar_selection_history", [])
        }
    
    def update_user_profile(self, profile_data: Dict[str, Any]) -> None:
        """Update user profile data"""
        if "gender" in profile_data:
            self.set_preference("user_profile.preferred_gender", profile_data["gender"])
        
        if "personality" in profile_data:
            # Clean and validate personality traits to prevent conflicts
            new_traits = profile_data["personality"]
            cleaned_traits = self._clean_personality_traits(new_traits)
            self.set_preference("user_profile.preferred_personality_traits", cleaned_traits)
        
        if "interaction_style" in profile_data:
            self.set_preference("user_profile.interaction_style", profile_data["interaction_style"])
    
    def _clean_personality_traits(self, traits: list) -> list:
        """Clean personality traits to prevent conflicts and limit to reasonable number"""
        if not traits:
            return []
        
        # Define conflicting trait pairs
        conflicts = {
            "energetic": ["calm", "steady"],
            "calm": ["energetic", "playful"],
            "playful": ["serious", "calm", "intellectual"],
            "intellectual": ["playful", "casual"],
            "mysterious": ["friendly", "open"],
            "friendly": ["mysterious", "reserved"],
            "sophisticated": ["casual", "simple"],
            "reliable": [],  # Doesn't conflict with others
        }
        
        cleaned = []
        for trait in traits:
            trait_lower = trait.lower()
            
            # Check if this trait conflicts with already added traits
            has_conflict = False
            for existing_trait in cleaned:
                existing_lower = existing_trait.lower()
                
                # Check both directions of conflict
                if (trait_lower in conflicts.get(existing_lower, []) or 
                    existing_lower in conflicts.get(trait_lower, [])):
                    has_conflict = True
                    break
            
            if not has_conflict:
                cleaned.append(trait)
        
        # Limit to maximum 3 traits to prevent over-specification
        return cleaned[:3]
    
    def get_app_settings(self) -> Dict[str, Any]:
        """Get application settings"""
        return self.get_preference("app_settings", {})
    
    def update_app_settings(self, settings: Dict[str, Any]) -> None:
        """Update application settings"""
        current_settings = self.get_app_settings()
        current_settings.update(settings)
        self.set_preference("app_settings", current_settings)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return self.get_preference("statistics", {})
    
    def reset_preferences(self) -> None:
        """Reset all preferences to defaults"""
        self.preferences = self.default_preferences.copy()
        self.save_preferences()
    
    def export_preferences(self, file_path: str) -> bool:
        """Export preferences to a file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.preferences, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error exporting preferences: {e}")
            return False
    
    def import_preferences(self, file_path: str) -> bool:
        """Import preferences from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                imported_prefs = json.load(f)
            
            # Validate and merge with current preferences
            self.preferences.update(imported_prefs)
            self.save_preferences()
            return True
        except Exception as e:
            print(f"Error importing preferences: {e}")
            return False


# Global instance for easy access
user_preferences = UserPreferences()


def get_user_preferences() -> UserPreferences:
    """Get the global user preferences instance"""
    return user_preferences


# Convenience functions
def should_show_welcome_screen() -> bool:
    """Check if welcome screen should be shown"""
    return user_preferences.should_show_welcome_screen()


def record_avatar_selection(avatar_name: str) -> None:
    """Record an avatar selection"""
    user_preferences.record_avatar_selection(avatar_name)
    user_preferences.save_preferences()


def get_last_selected_avatar() -> Optional[str]:
    """Get the last selected avatar"""
    return user_preferences.get_last_selected_avatar()


def get_user_profile_for_recommendation() -> Dict[str, Any]:
    """Get user profile for recommendation"""
    return user_preferences.get_user_profile_for_recommendation()


# Example usage and testing
if __name__ == "__main__":
    # Test the preferences system
    prefs = UserPreferences()
    
    print("Testing User Preferences System")
    print("=" * 40)
    
    # Test basic operations
    print(f"Should show welcome screen: {prefs.should_show_welcome_screen()}")
    print(f"Last selected avatar: {prefs.get_last_selected_avatar()}")
    
    # Test recording selections
    prefs.record_avatar_selection("Amelia")
    prefs.record_avatar_selection("Gura")
    prefs.record_avatar_selection("Amelia")
    
    print(f"After selections - Last avatar: {prefs.get_last_selected_avatar()}")
    print(f"Statistics: {prefs.get_statistics()}")
    
    # Save preferences
    prefs.save_preferences()
    print("Preferences saved successfully!")
