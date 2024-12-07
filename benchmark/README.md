# VTuber LLM Benchmark System

## Overview
This benchmark system evaluates how well a Language Model (LLM) can maintain the personality, characteristics, and behavior of a VTuber character (in this case, Amelia Watson). The system uses multiple metrics to assess the model's performance across different streaming scenarios and interactions.

## Core Components

### 1. Model Initialization
```python
def __init__(self, model_path="bluuwhale/L3-SthenoMaidBlackroot-8B-V1"):
    # Initialize models and configurations
```
- Uses 4-bit quantization for efficient model loading
- Loads sentiment analyzer for emotional evaluation
- Initializes ROUGE scorer for reference similarity
- Loads character profile and test cases

### 2. Evaluation Metrics

#### 2.1 Personality Consistency (0-100%)
Measures how well the response matches the character's established personality traits:
- Weighted trait system for different aspects of personality
- Direct matches (exact terms) and contextual matches (related phrases)
- Speech pattern bonuses for characteristic expressions
- Core traits (detective, time traveler) weighted higher than secondary traits

#### 2.2 Topic Relevance (0-100%)
Evaluates if the response stays on topic and matches common VTuber topics:
- Gaming content
- Streaming activities
- Character-specific topics (investigations, time travel)
- Fan interactions

#### 2.3 Emotional Consistency (0-100%)
Assesses if the emotional tone matches the expected emotion:
- Uses BERT-based sentiment analysis
- Compares with character's emotional tendencies
- Considers context-appropriate emotions

#### 2.4 Reference Similarity (0-100%)
Measures response quality and character-specific elements:
- Base score for substantial responses
- Bonuses for using character speech patterns
- Bonuses for covering relevant topics

### 3. Test Cases
The benchmark includes diverse test scenarios:
1. Stream start
2. Gaming/FPS moments
3. Superchat reading
4. Storytelling
5. Investigation streams
6. Time travel stories
7. Fan interactions
8. Character-specific content

### 4. Character Profile
```json
{
  "personality_traits": [...],
  "speech_patterns": [...],
  "common_topics": [...],
  "emotional_tendencies": {...}
}
```
- Defines character's core traits
- Lists speech patterns and expressions
- Specifies common topics
- Maps emotional tendencies

## Usage

### Running the Benchmark
```bash
python benchmark/benchmark_LLM.py
```

### Output Format
```
Benchmark Results:
==================================================

Average Metrics:
--------------------------------------------------
Metric                          Score
--------------------------------------------------
Personality Consistency         23.08%
Topic Relevance                10.71%
Emotional Consistency          12.24%
Reference Similarity           75.00%

Detailed Results:
[Individual test results with metrics]
```

## Technical Details

### Dependencies
- transformers
- torch
- rouge_score
- bert_score
- numpy

### Model Configuration
- 4-bit quantization using BitsAndBytesConfig
- Uses device_map="auto" for optimal resource allocation
- Implements proper token handling and generation parameters

### Generation Parameters
```python
outputs = self.model.generate(
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.15
)
```

## Scoring System

### Personality Consistency
```python
personality_indicators = {
    "trait": {
        "direct": [...],    # Full score terms
        "contextual": [...], # Partial score terms
        "weight": float     # Trait importance
    }
}
```

### Topic Relevance
- Direct topic matches
- Related keyword matches
- Context consideration
- Weighted scoring system

### Emotional Consistency
- BERT-based sentiment analysis
- Emotion distribution matching
- Context-aware evaluation

## Customization

### Adding New Test Cases
```python
test_cases = [
    {
        "context": str,
        "prompt": str,
        "expected_emotion": str,
        "expected_topics": list,
        "expected_traits": list
    }
]
```

### Modifying Character Profile
- Update personality_traits
- Adjust speech_patterns
- Modify common_topics
- Tune emotional_tendencies

## Best Practices

1. **Test Case Design**
   - Cover diverse scenarios
   - Include character-specific situations
   - Test different emotional states
   - Vary interaction types

2. **Metric Tuning**
   - Balance weights for different traits
   - Adjust scoring thresholds
   - Fine-tune pattern matching
   - Calibrate emotional evaluation

3. **Response Evaluation**
   - Consider context
   - Check for character consistency
   - Evaluate natural speech patterns
   - Assess topic relevance

## Future Improvements

1. **Enhanced Metrics**
   - More nuanced personality evaluation
   - Better context understanding
   - Improved emotion detection

2. **Additional Features**
   - Multi-character support
   - Interactive testing mode
   - Detailed analysis reports

3. **Model Optimization**
   - Fine-tuning suggestions
   - Performance improvements
   - Memory optimization 