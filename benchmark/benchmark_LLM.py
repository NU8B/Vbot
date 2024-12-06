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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import random

class VTuberLLMBenchmark:
    def __init__(self, model_path="facebook/opt-125m"):
        # Load the processed data
        self.character_profile = self._load_profile()
        self.reference_conversations = self._load_conversations()
        
        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Load sentiment analyzer
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        
        # Load LLM model
        print("Loading LLM model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=False,
            torch_dtype=torch.float16
        )
    
    def _load_profile(self):
        """Load the character profile from processed data"""
        with open("benchmark/data/character_profile.json", 'r') as f:
            return json.load(f)
    
    def _load_conversations(self):
        """Load the processed conversations"""
        with open("benchmark/data/processed_conversations.json", 'r') as f:
            return json.load(f)
    
    def evaluate_personality_consistency(self, generated_text: str) -> float:
        """Evaluate how well the response matches the character's personality"""
        score = 0
        total_traits = len(self.character_profile["personality_traits"])
        
        # Check for personality traits
        for trait in self.character_profile["personality_traits"]:
            if trait.lower() in generated_text.lower():
                score += 1
        
        # Normalize score
        return score / total_traits if total_traits > 0 else 0
    
    def evaluate_topic_relevance(self, context: str, generated_text: str) -> float:
        """Evaluate if the response stays on topic and matches common topics"""
        score = 0
        total_topics = len(self.character_profile["common_topics"])
        
        # Check for topic relevance
        for topic in self.character_profile["common_topics"]:
            if topic.lower() in generated_text.lower() or topic.lower() in context.lower():
                score += 1
        
        return score / total_topics if total_topics > 0 else 0
    
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
        generated_response = self.generate_response(prompt)
        
        # Calculate metrics
        personality_score = self.evaluate_personality_consistency(generated_response)
        topic_score = self.evaluate_topic_relevance(context, generated_response)
        emotion_score = self.evaluate_emotional_consistency(generated_response, expected_emotion)
        
        # Find most similar reference conversation for comparison
        best_rouge_score = 0
        reference_response = ""
        
        for conv in self.reference_conversations:
            if conv["prompt"].lower() in prompt.lower() or prompt.lower() in conv["prompt"].lower():
                scores = self.rouge_scorer.score(conv["response"], generated_response)
                rouge_l = scores['rougeL'].fmeasure
                if rouge_l > best_rouge_score:
                    best_rouge_score = rouge_l
                    reference_response = conv["response"]
        
        return {
            "prompt": prompt,
            "generated_response": generated_response,
            "reference_response": reference_response,
            "metrics": {
                "personality_consistency": personality_score,
                "topic_relevance": topic_score,
                "emotional_consistency": emotion_score,
                "reference_similarity": best_rouge_score
            }
        }
    
    def run_benchmark(self, num_samples: int = 10) -> Dict:
        """Run benchmark on multiple samples"""
        results = []
        avg_metrics = {
            "personality_consistency": 0,
            "topic_relevance": 0,
            "emotional_consistency": 0,
            "reference_similarity": 0
        }
        
        # Sample from reference conversations
        sample_convs = random.sample(self.reference_conversations, min(num_samples, len(self.reference_conversations)))
        
        for conv in sample_convs:
            result = self.benchmark_response(
                prompt=conv["prompt"],
                context=conv["context"],
                expected_emotion=conv["emotion"]
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
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using the LLM"""
        # Prepare the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15
            )
        
        # Decode and return
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()

def main():
    # Initialize benchmark
    benchmark = VTuberLLMBenchmark()
    
    # Run benchmark with 5 samples
    results = benchmark.run_benchmark(num_samples=5)
    
    # Print results
    print("\nBenchmark Results:")
    print("=================")
    print("\nAverage Metrics:")
    for metric, value in results["average_metrics"].items():
        print(f"{metric}: {value:.3f}")
    
    print("\nSample Results:")
    for i, result in enumerate(results["individual_results"][:2]):
        print(f"\nTest {i+1}:")
        print(f"Prompt: {result['prompt']}")
        print(f"Generated: {result['generated_response'][:100]}...")
        print(f"Metrics: {result['metrics']}")

if __name__ == "__main__":
    main()