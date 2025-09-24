# Vbot Welcome Screen & Avatar Recommender System

## Overview

The Vbot application now features a comprehensive welcome screen and avatar recommendation system that enhances the user experience by providing personalized avatar suggestions and a modern interface for avatar selection.

## New Features

### 1. Welcome Screen
- **Modern UI**: Clean, dark-themed interface with responsive design
- **Avatar Selection**: Grid-based layout showcasing all available avatars
- **Avatar Information**: Detailed descriptions, personality traits, and voice characteristics
- **User-Friendly**: Hover effects, clickable cards, and intuitive navigation

### 2. Avatar Recommendation System
- **Intelligent Recommendations**: AI-powered suggestions based on user preferences
- **Personality Matching**: Matches user preferences with avatar characteristics
- **Gender Preferences**: Supports gender-based filtering
- **Usage History**: Learns from user behavior to improve recommendations
- **Variety Encouragement**: Suggests different avatars to promote exploration

### 3. User Preferences Management
- **Persistent Storage**: Saves user preferences and selection history
- **Smart Defaults**: Remembers last selected avatar for returning users
- **First-Time Experience**: Special handling for new users
- **Statistics Tracking**: Monitors usage patterns and favorite avatars

## Available Avatars

### Amelia Watson
- **Personality**: Detective, Curious, Energetic
- **Voice**: Bright and cheerful
- **Best For**: Users who enjoy mystery and adventure

### Eveland Novice
- **Personality**: Calm, Intellectual, Sophisticated
- **Voice**: Deep and thoughtful
- **Best For**: Users seeking wisdom and thoughtful conversations

### Gawr Gura
- **Personality**: Playful, Friendly, Mischievous
- **Voice**: Cute and playful
- **Best For**: Casual conversations and fun interactions

### Shiori Novella
- **Personality**: Mysterious, Artistic, Thoughtful
- **Voice**: Soft and mysterious
- **Best For**: Users interested in creativity and depth

### Wilson
- **Personality**: Reliable, Supportive, Steady
- **Voice**: Warm and steady
- **Best For**: Users seeking guidance and support

## How It Works

### First-Time Users
1. Launch Vbot application
2. Welcome screen appears automatically
3. Browse available avatars or use recommendation system
4. Select preferred avatar to continue
5. Preferences are saved for future sessions

### Returning Users
1. Application launches with last selected avatar
2. Welcome screen can be accessed via background menu
3. Avatar switching available through main interface
4. Recommendations improve based on usage history

### Command Line Options
```bash
# Launch with specific avatar (skips welcome screen)
python Vbot.py --model Amelia

# Skip welcome screen entirely (uses default)
python Vbot.py --skip-welcome

# Force welcome screen (default behavior for new users)
python Vbot.py
```

## Technical Implementation

### Modular Architecture
- **`utils/welcome_screen.py`**: Main welcome screen implementation
- **`utils/user_preferences.py`**: User preferences and data management
- **Integration**: Seamlessly integrated with existing Vbot.py

### Key Components
1. **AvatarRecommender**: Handles recommendation logic
2. **WelcomeScreen**: Main UI interface
3. **RecommendationDialog**: Interactive preference collection
4. **UserPreferences**: Data persistence and management

### Recommendation Algorithm
The system uses a scoring algorithm that considers:
- Gender preferences (weight: 2)
- Personality trait matching (weight: 1 per match)
- Favorite avatar bonus (weight: 3)
- Recent usage penalty (weight: -1 for variety)

## Configuration

### User Preferences Location
- **Windows**: `%USERPROFILE%\.vbot\user_preferences.json`
- **Configuration**: Automatically created on first use
- **Backup**: Can be exported/imported for backup purposes

### Customization Options
- Show/hide welcome screen preference
- Avatar selection history (last 10 selections)
- User profile data for recommendations
- Application settings and statistics

## Testing

### Test Scripts
- **`test_recommender_only.py`**: Tests recommendation system
- **`test_welcome_screen.py`**: Full UI testing (requires display)

### Running Tests
```bash
# Test recommendation system
python test_recommender_only.py

# Test full welcome screen (GUI required)
python test_welcome_screen.py
```

## Future Enhancements

### Planned Features
1. **Avatar Images**: Replace placeholders with actual avatar images
2. **Advanced ML**: Machine learning-based recommendations
3. **Custom Avatars**: User-created avatar support
4. **Voice Previews**: Audio samples in selection screen
5. **Themes**: Multiple UI themes and customization

### Extensibility
The modular design allows for easy extension:
- Add new avatars by updating `avatar_profiles`
- Enhance recommendation algorithm
- Customize UI themes and layouts
- Integrate with external avatar systems

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Display Issues**: Check tkinter installation for GUI
3. **Preferences Reset**: Delete `.vbot` folder to reset all preferences

### Debug Mode
Enable debug output by setting environment variable:
```bash
set VBOT_DEBUG=1
python Vbot.py
```

## Contributing

When adding new avatars or features:
1. Update `avatar_profiles` in `welcome_screen.py`
2. Add corresponding model files in `asset/model/`
3. Update documentation and tests
4. Ensure backward compatibility

## License

This welcome screen system is part of the Vbot project and follows the same licensing terms as the main application.
