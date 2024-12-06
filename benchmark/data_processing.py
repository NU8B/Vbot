from typing import List, Dict
import re
from pathlib import Path
import json

def clean_transcript(text: str) -> List[Dict]:
    """Process raw transcript into clean conversation pairs."""
    # Split by [Music] and clean up segments
    segments = text.split("[Music]")
    
    conversations = []
    current_context = "Stream announcement follow-up"
    current_response = []
    current_prompt = None
    
    # Common emotion indicators with their weights
    emotion_indicators = {
        'laugh': ('amused', 1),
        'thank': ('grateful', 0.8),
        'sorry': ('apologetic', 0.9),
        'kidding': ('playful', 1),
        'excited': ('excited', 1),
        'sad': ('sad', 0.9),
        'nervous': ('nervous', 0.8),
        'happy': ('happy', 1),
        'glad': ('happy', 0.8),
        'appreciate': ('grateful', 0.9),
        'love': ('affectionate', 1),
        'miss': ('nostalgic', 0.9),
        'hope': ('hopeful', 0.8)
    }
    
    def detect_emotion(text: str) -> str:
        max_confidence = 0
        emotion = "neutral"
        for indicator, (emotion_type, weight) in emotion_indicators.items():
            if indicator in text.lower():
                if weight > max_confidence:
                    emotion = emotion_type
                    max_confidence = weight
        return emotion
    
    def save_conversation():
        nonlocal current_prompt, current_response, conversations
        if current_prompt and current_response:
            response_text = " ".join(current_response)
            if len(response_text.strip()) > 10:  # Ensure meaningful response
                conversations.append({
                    "context": current_context,
                    "prompt": current_prompt,
                    "response": response_text,
                    "emotion": detect_emotion(response_text)
                })
            current_response = []
            current_prompt = None
    
    for segment in segments:
        lines = [line.strip() for line in segment.split('\n') if line.strip()]
        
        for line in lines:
            # Skip technical lines and markdown
            if any(marker in line for marker in ['```', '<', '>', 'Human:', 'Assistant:', 'test test']):
                continue
                
            # Detect context changes (longer explanatory lines)
            if len(line) > 100 and not current_prompt:
                current_context = line[:100] + "..."
                save_conversation()
                continue
            
            # Detect questions or superchats
            if ('?' in line or 'thank you' in line.lower()) and len(line) < 100:
                save_conversation()
                current_prompt = line
                continue
            
            # Collect response
            if current_prompt and len(line) > 5:
                current_response.append(line)
            
            # If line indicates a topic change or new segment
            if len(line) > 50 and not current_prompt and not any(marker in line.lower() for marker in ['thank you', '?']):
                save_conversation()
                current_context = line[:50] + "..."
    
    # Save final conversation if exists
    save_conversation()
    
    return conversations

def save_processed_data(conversations: List[Dict], output_file: str = "processed_conversations.json"):
    """Save processed conversations to JSON file"""
    output_path = Path("benchmark/data") / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)

def extract_character_profile(conversations: List[Dict]) -> Dict:
    """
    Extract character profile from conversations
    """
    profile = {
        "personality_traits": set(),
        "speech_patterns": set(),
        "common_topics": set(),
        "emotional_tendencies": {}
    }
    
    # Personality trait indicators
    trait_indicators = {
        "kidding": "playful",
        "sorry": "apologetic",
        "thank you": "grateful",
        "excited": "enthusiastic",
        "maybe": "thoughtful",
        "I think": "analytical",
        "I'm sure": "confident",
        "I don't know": "honest",
        "let me": "helpful",
        "I hope": "optimistic",
        "I will": "determined"
    }
    
    # Topic indicators
    topic_indicators = {
        "game": "gaming",
        "stream": "streaming",
        "community": "community",
        "teammates": "community",
        "project": "projects",
        "music": "entertainment",
        "sing": "entertainment",
        "minecraft": "gaming",
        "charity": "charity work",
        "collab": "collaborations"
    }
    
    for conv in conversations:
        response = conv["response"].lower()
        emotion = conv["emotion"]
        
        # Track emotional tendencies
        profile["emotional_tendencies"][emotion] = profile["emotional_tendencies"].get(emotion, 0) + 1
        
        # Extract personality traits
        for indicator, trait in trait_indicators.items():
            if indicator in response:
                profile["personality_traits"].add(trait)
        
        # Extract speech patterns (recurring phrases)
        common_phrases = [
            phrase.strip('" ')
            for phrase in re.findall(r'"([^"]*)"', response)
            if len(phrase.strip()) > 3
        ]
        profile["speech_patterns"].update(common_phrases)
        
        # Extract common topics
        for indicator, topic in topic_indicators.items():
            if indicator in response:
                profile["common_topics"].add(topic)
    
    # Convert sets to lists for JSON serialization
    return {
        "personality_traits": list(profile["personality_traits"]),
        "speech_patterns": list(profile["speech_patterns"]),
        "common_topics": list(profile["common_topics"]),
        "emotional_tendencies": profile["emotional_tendencies"]
    }

def main():
    # Read transcript
    transcript_path = Path("asset/ame-vods-transcripts/chat-stream.txt")
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = f.read()
    
    print("Processing transcript...")
    print(f"Total lines: {len(transcript.splitlines())}")
    
    # Process conversations
    conversations = clean_transcript(transcript)
    
    print("\nExample conversations:")
    for i, conv in enumerate(conversations[:5]):  # Show first 5 conversations
        print(f"\nConversation {i+1}:")
        print(f"Context: {conv['context']}")
        print(f"Prompt: {conv['prompt']}")
        print(f"Response: {conv['response'][:100]}...")
        print(f"Emotion: {conv['emotion']}")
    
    # Save processed conversations
    save_processed_data(conversations)
    
    # Extract and save character profile
    profile = extract_character_profile(conversations)
    save_processed_data(profile, "character_profile.json")
    
    print(f"\nProcessed {len(conversations)} conversations")
    print("\nCharacter Profile Preview:")
    print(json.dumps(profile, indent=2))

if __name__ == "__main__":
    main()