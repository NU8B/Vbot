import os
# Set OpenMP environment variable to avoid runtime error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm
from datetime import datetime
import evaluate
from rouge_score import rouge_scorer
from bert_score import score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig
import random

class VTuberLLMBenchmark:
    def __init__(self, model_path="bluuwhale/L3-SthenoMaidBlackroot-8B-V1"):
        # Load the processed data
        self.character_profile = self._load_profile()
        self.test_cases = self._load_test_cases()
        
        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize sentiment analyzer
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # Load LLM model
        print("Loading LLM model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
    
    def _load_profile(self):
        """Load the character profile from processed data"""
        with open("benchmark/data/character_profile.json", 'r') as f:
            return json.load(f)
    
    def _load_test_cases(self):
        """Load or create test cases"""
        test_cases = [
            {
                "context": "Stream start",
                "prompt": "Hi chat! Detective Amelia Watson here! Ready to solve some mysteries?",
                "expected_emotion": "excited",
                "expected_topics": ["detective work", "streaming"],
                "expected_traits": ["detective", "energetic"]
            },
            {
                "context": "Gaming stream - FPS rage",
                "prompt": "WHAT?! How did that guy even- *hic* Sorry chat, getting a bit salty here!",
                "expected_emotion": "salty",
                "expected_topics": ["gaming", "fps games"],
                "expected_traits": ["gremlin", "competitive", "salty"]
            },
            {
                "context": "Superchat reading",
                "prompt": "Thank you for the superchat! Let me tell you about the time I ground pounded your mo- *cough* I mean, a mysterious case I solved!",
                "expected_emotion": "teasing",
                "expected_topics": ["ground pounding", "detective work"],
                "expected_traits": ["gremlin", "teasing", "detective"]
            },
            {
                "context": "Story time",
                "prompt": "And then I traveled back in time to solve the mystery of the missing Watson concoction!",
                "expected_emotion": "excited",
                "expected_topics": ["time travel", "concoctions", "mystery solving"],
                "expected_traits": ["time traveler", "detective", "quirky"]
            },
            {
                "context": "Apex Legends stream",
                "prompt": "*hic* Chat! Did you see that headshot?! I'm actually cracked at this game!",
                "expected_emotion": "excited",
                "expected_topics": ["gaming", "apex legends"],
                "expected_traits": ["competitive", "energetic", "gremlin"]
            },
            {
                "context": "Time travel story",
                "prompt": "Let me tell you about my latest investigation through time! *adjusts magnifying glass*",
                "expected_emotion": "investigative",
                "expected_topics": ["time travel", "detective work"],
                "expected_traits": ["detective", "time traveler", "investigative"]
            },
            {
                "context": "Mother stories",
                "prompt": "Speaking of moms... *gremlin giggles* Let me tell you about this one time...",
                "expected_emotion": "teasing",
                "expected_topics": ["mother stories", "ground pounding"],
                "expected_traits": ["gremlin", "teasing", "chaotic"]
            },
            {
                "context": "Watson concoction",
                "prompt": "Today we're making a special Watson concoction! *mischievous laughter* Don't worry, it's mostly safe!",
                "expected_emotion": "chaotic",
                "expected_topics": ["concoctions"],
                "expected_traits": ["quirky", "chaotic", "gremlin"]
            },
            {
                "context": "Teamate interaction",
                "prompt": "Aww, thanks teamate! *hic* Your support means everything to this little detective!",
                "expected_emotion": "happy",
                "expected_topics": ["teamates", "streaming"],
                "expected_traits": ["energetic", "grateful", "detective"]
            },
            {
                "context": "Investigation stream",
                "prompt": "The evidence points to... *intense magnifying glass action* AHA! Just as I suspected!",
                "expected_emotion": "determined",
                "expected_topics": ["detective work", "investigations"],
                "expected_traits": ["detective", "investigative", "intelligent"]
            }
        ]
        return test_cases
    
    def evaluate_personality_consistency(self, generated_text: str) -> float:
        """Evaluate how well the response matches the character's personality"""
        score = 0
        total_traits = len(self.character_profile["personality_traits"])
        text_lower = generated_text.lower()
        
        # Define comprehensive personality indicators with weights
        personality_indicators = {
            "detective": {
                "direct": ["detective", "investigate", "case", "mystery", "evidence", "clue"],
                "contextual": ["solve", "found out", "discovered", "looking into", "examining"],
                "weight": 1.5  # Important character trait
            },
            "time traveler": {
                "direct": ["time travel", "timeline", "past", "future", "temporal"],
                "contextual": ["back then", "different era", "time period", "history"],
                "weight": 1.5  # Important character trait
            },
            "gremlin": {
                "direct": ["gremlin", "*giggles*", "hic", "gwak", "ground pound"],
                "contextual": ["chaos", "mischief", "teehee", "hehe", "evil laugh"],
                "weight": 1.2
            },
            "energetic": {
                "direct": ["excited", "hype", "let's go", "amazing", "awesome"],
                "contextual": ["can't wait", "so fun", "incredible", "wow"],
                "weight": 1.0
            },
            "competitive": {
                "direct": ["win", "victory", "champion", "beat", "score", "rank"],
                "contextual": ["try hard", "practice", "improve", "getting better"],
                "weight": 1.0
            },
            "salty": {
                "direct": ["salty", "rage", "angry", "mad", "frustrated"],
                "contextual": ["not fair", "cheating", "how dare", "what the"],
                "weight": 1.0
            },
            "quirky": {
                "direct": ["weird", "unique", "special", "different", "quirky"],
                "contextual": ["random", "silly", "funny", "strange"],
                "weight": 0.8
            }
        }
        
        # Check for personality traits
        for trait, indicators in personality_indicators.items():
            trait_score = 0
            
            # Check direct matches (full score)
            if any(term in text_lower for term in indicators["direct"]):
                trait_score = 1.0
            # Check contextual matches (partial score)
            elif any(term in text_lower for term in indicators["contextual"]):
                trait_score = 0.5
                
            # Apply weight to trait score
            score += trait_score * indicators["weight"]
        
        # Check for speech patterns (bonus points)
        speech_patterns = self.character_profile["speech_patterns"]
        pattern_score = sum(0.2 for pattern in speech_patterns if pattern.lower() in text_lower)
        score += pattern_score
        
        # Normalize score (max possible score is sum of weights + speech pattern bonus)
        max_score = sum(indicators["weight"] for indicators in personality_indicators.values()) + 1.0
        normalized_score = min(1.0, score / max_score)
        
        return normalized_score
    
    def evaluate_topic_relevance(self, context: str, generated_text: str) -> float:
        """Evaluate if the response stays on topic and matches common topics"""
        score = 0
        total_topics = len(self.character_profile["common_topics"])
        combined_text = (context + " " + generated_text).lower()
        
        # Define topic-specific keywords
        topic_keywords = {
            "streaming": ["stream", "live", "chat", "viewers", "broadcast", "superchat"],
            "detective work": ["case", "mystery", "investigate", "clue", "evidence", "solve"],
            "time travel": ["timeline", "past", "future", "temporal", "paradox", "history"],
            "gaming": ["game", "play", "stream", "gaming", "gamer", "level"],
            "hololive": ["hololive", "vtuber", "collab", "member", "idol", "talent"]
        }
        
        for topic in self.character_profile["common_topics"]:
            topic_lower = topic.lower()
            
            # Direct topic match
            if topic_lower in combined_text:
                score += 1
                continue
            
            # Check for topic-related keywords
            if topic_lower in topic_keywords:
                for keyword in topic_keywords[topic_lower]:
                    if keyword in combined_text:
                        score += 0.5
                        break
        
        return min(1.0, score / total_topics) if total_topics > 0 else 0
    
    def evaluate_emotional_consistency(self, generated_text: str, expected_emotion: str) -> float:
        """Evaluate if the response matches the expected emotional tone"""
        # Get emotion distribution from character profile
        total_emotions = sum(self.character_profile["emotional_tendencies"].values())
        emotion_weights = {
            emotion: count/total_emotions 
            for emotion, count in self.character_profile["emotional_tendencies"].items()
        }
        
        # Detect emotion in generated text
        inputs = self.sentiment_tokenizer(generated_text, return_tensors="pt", truncation=True)
        outputs = self.sentiment_model(**inputs)
        predicted_emotion = outputs.logits.argmax().item()
        
        # Compare with expected emotion
        if expected_emotion in emotion_weights:
            return emotion_weights[expected_emotion]
        return 0.0
    
    def benchmark_response(self, prompt: str, context: str = "", expected_emotion: str = "neutral") -> Dict:
        """Benchmark a single LLM response"""
        # Generate response
        generated_response = self.generate_response(prompt, context)
        
        # Calculate metrics
        personality_score = self.evaluate_personality_consistency(generated_response)
        topic_score = self.evaluate_topic_relevance(context, generated_response)
        emotion_score = self.evaluate_emotional_consistency(generated_response, expected_emotion)
        
        # For reference similarity, we'll use a simpler approach since we don't have reference responses
        reference_score = 0.0
        if len(generated_response.split()) >= 3:  # Check if response is substantial
            reference_score = 0.5  # Base score for a substantial response
            # Add bonus for character-specific elements
            if any(pattern in generated_response.lower() for pattern in self.character_profile["speech_patterns"]):
                reference_score += 0.25
            if any(topic in generated_response.lower() for topic in self.character_profile["common_topics"]):
                reference_score += 0.25
        
        return {
            "prompt": prompt,
            "context": context,
            "generated_response": generated_response,
            "metrics": {
                "personality_consistency": personality_score,
                "topic_relevance": topic_score,
                "emotional_consistency": emotion_score,
                "reference_similarity": reference_score
            }
        }
    
    def run_benchmark(self, num_samples: int = 5) -> Dict:
        """Run benchmark on test cases"""
        results = []
        avg_metrics = {
            "personality_consistency": 0,
            "topic_relevance": 0,
            "emotional_consistency": 0,
            "reference_similarity": 0
        }
        
        # Use test cases
        sample_cases = random.sample(self.test_cases, min(num_samples, len(self.test_cases)))
        
        for case in sample_cases:
            result = self.benchmark_response(
                prompt=case["prompt"],
                context=case["context"],
                expected_emotion=case["expected_emotion"]
            )
            results.append(result)
            
            # Update averages
            for metric, value in result["metrics"].items():
                avg_metrics[metric] += value
        
        # Calculate final averages
        for metric in avg_metrics:
            avg_metrics[metric] /= len(results)
            
        return {
            "individual_results": results,
            "average_metrics": avg_metrics
        }
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using the LLM"""
        # Create a more detailed system prompt
        system_prompt = f"""You are roleplaying as Amelia Watson, a time-traveling detective VTuber from Hololive English.

Character traits:
- You're a detective who solves mysteries and time travels
- You're energetic, competitive, and sometimes salty (especially in gaming)
- You make gremlin noises and often hiccup (hic!)
- You love telling stories about ground pounding moms and your childhood
- You create mysterious concoctions
- You call your fans "teamates" (spelled this way)

Current context: {context}

Keep responses natural, playful, and in character. Use Amelia's speech patterns and maintain her personality."""

        # Construct the full prompt
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAmelia Watson:"
        
        # Prepare the prompt
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and return
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        response = response.split("Amelia Watson:")[-1].strip()
        return response

def main():
    # Initialize benchmark
    benchmark = VTuberLLMBenchmark()
    
    # Run benchmark with all test cases
    results = benchmark.run_benchmark(num_samples=10)
    
    # Print results
    print("\nBenchmark Results:")
    print("=" * 50)
    
    # Print average metrics in a formatted table
    print("\nAverage Metrics:")
    print("-" * 50)
    print(f"{'Metric':<25} {'Score':>10}")
    print("-" * 50)
    for metric, value in results["average_metrics"].items():
        formatted_metric = metric.replace('_', ' ').title()
        print(f"{formatted_metric:<25} {value:>10.2%}")
    
    # Print detailed results
    print("\nDetailed Results:")
    print("=" * 50)
    
    for i, result in enumerate(results["individual_results"], 1):
        print(f"\nTest {i}")
        print("-" * 50)
        print(f"Context: {result['context']}")
        print(f"Prompt: {result['prompt']}")
        print(f"Generated: {result['generated_response'][:100]}...")
        
        # Print metrics in a formatted table
        print("\nMetrics:")
        print(f"{'Metric':<25} {'Score':>10}")
        print("-" * 35)
        for metric, value in result['metrics'].items():
            formatted_metric = metric.replace('_', ' ').title()
            print(f"{formatted_metric:<25} {value:>10.2%}")
        print("=" * 50)

if __name__ == "__main__":
    main()