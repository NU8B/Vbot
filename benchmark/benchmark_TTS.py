import torch
import torchaudio
import numpy as np
from pesq import pesq
from scipy.spatial.distance import cosine
from resemblyzer import VoiceEncoder
import librosa
import json
from pathlib import Path
from tqdm import tqdm
from faster_whisper import WhisperModel
from datetime import datetime
import os
import sys

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))
from utils.inference_styleTTS2 import StyleTTS2Inference


class VoiceBenchmark:
    def __init__(self, model_path=None):
        self.tts = (
            StyleTTS2Inference(repo_id=model_path)
            if model_path
            else StyleTTS2Inference()
        )
        self.voice_encoder = VoiceEncoder()
        self.whisper_model = WhisperModel(
            "distil-large-v3",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "float32",
        )

    def extract_speaker_embedding(self, audio_path):
        """Extract speaker embedding using Resemblyzer"""
        # Load at 24000Hz to match StyleTTS2
        wav, sr = librosa.load(audio_path, sr=24000)
        # Resample to 16000Hz for Resemblyzer (it expects 16000Hz)
        wav = librosa.resample(wav, orig_sr=24000, target_sr=16000)
        embedding = self.voice_encoder.embed_utterance(wav)
        return embedding

    def compute_speaker_similarity(self, embedding1, embedding2):
        """Compute cosine similarity between speaker embeddings"""
        return 1 - cosine(embedding1, embedding2)

    def compute_pesq_score(self, reference_path, generated_path):
        """Compute PESQ score between reference and generated audio"""
        # Load audio at 24000Hz to match StyleTTS2's output
        ref_wav, sr = librosa.load(reference_path, sr=24000)
        gen_wav, _ = librosa.load(generated_path, sr=24000)

        # Ensure both audio files have the same length
        min_len = min(len(ref_wav), len(gen_wav))
        ref_wav = ref_wav[:min_len]
        gen_wav = gen_wav[:min_len]

        # Resample to 16000Hz for PESQ calculation (PESQ requirement)
        ref_wav = librosa.resample(ref_wav, orig_sr=24000, target_sr=16000)
        gen_wav = librosa.resample(gen_wav, orig_sr=24000, target_sr=16000)

        try:
            score = pesq(16000, ref_wav, gen_wav, "nb")
            return score
        except Exception as e:
            print(f"PESQ calculation failed: {str(e)}")
            return None

    def transcribe_audio(self, audio_path):
        """Transcribe audio using faster-whisper"""
        segments, _ = self.whisper_model.transcribe(
            audio_path,
            beam_size=5,
            language="en",
            condition_on_previous_text=False,
        )
        # Combine all segments into one text
        return " ".join(segment.text.strip() for segment in segments).lower()

    def compute_transcription_accuracy(self, original_text, generated_text):
        """Compute word-level accuracy between original and generated transcriptions"""
        original_words = set(original_text.split())
        generated_words = set(generated_text.split())

        if not original_words:
            return 0.0

        common_words = original_words.intersection(generated_words)
        return len(common_words) / len(original_words)

    def benchmark_sample(self, reference_path, text, output_path="generated.wav"):
        """Benchmark a single sample"""
        # Generate speech
        ref_s = self.tts.compute_style(reference_path)
        generated_wav = self.tts.inference(
            text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=10, embedding_scale=1
        )

        # Save generated audio
        import soundfile as sf

        sf.write(output_path, generated_wav, 24000)

        # Extract speaker embeddings
        ref_embedding = self.extract_speaker_embedding(reference_path)
        gen_embedding = self.extract_speaker_embedding(output_path)

        # Compute speaker similarity
        speaker_similarity = self.compute_speaker_similarity(
            ref_embedding, gen_embedding
        )

        # Compute PESQ score
        pesq_score = self.compute_pesq_score(reference_path, output_path)

        # Compute transcription accuracy
        original_transcription = self.transcribe_audio(reference_path)
        generated_transcription = self.transcribe_audio(output_path)
        transcription_accuracy = self.compute_transcription_accuracy(
            original_transcription, generated_transcription
        )

        return {
            "speaker_similarity": float(speaker_similarity),
            "pesq_score": float(pesq_score) if pesq_score else None,
            "transcription_accuracy": float(transcription_accuracy),
            "original_transcription": original_transcription,
            "generated_transcription": generated_transcription,
        }

    def benchmark_dataset(self, dataset_path, output_dir="results"):
        """
        Benchmark multiple samples from a dataset
        Dataset structure should be:
        dataset_path/
            - sample1.wav
            - sample2.wav
            ...
        Results will be saved in:
        results/
            YYYY-MM-DD_HH-MM-SS/
                - generated_samples/
                    - generated_sample1.wav
                    - generated_sample2.wav
                    ...
                - benchmark_results.json
                - summary.txt
        """
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_output_dir = Path(output_dir)
        output_dir = base_output_dir / timestamp
        samples_dir = output_dir / "generated_samples"

        # Create directory structure
        output_dir.mkdir(parents=True, exist_ok=True)
        samples_dir.mkdir(exist_ok=True)

        dataset_path = Path("Data_prep/raw_data/raw_audio2")
        wav_files = list(dataset_path.glob("*.wav"))

        print(f"Found {len(wav_files)} WAV files in {dataset_path}")
        if len(wav_files) == 0:
            print("No WAV files found! Please check the directory path.")
            return None

        results = []
        avg_metrics = {
            "speaker_similarity": 0,
            "pesq_score": 0,
            "transcription_accuracy": 0,
        }
        valid_pesq_scores = 0
        processed_files = 0

        for wav_file in tqdm(wav_files, desc="Benchmarking samples"):
            try:
                # Transcribe the original audio
                print(f"\nTranscribing {wav_file.name}...")
                segments, _ = self.whisper_model.transcribe(
                    str(wav_file),
                    beam_size=5,
                    language="en",
                    condition_on_previous_text=False,
                )
                text = " ".join(segment.text.strip() for segment in segments)

                if not text:
                    print(f"Warning: Could not transcribe {wav_file.name}")
                    continue

                # Generate and evaluate synthetic speech
                output_path = samples_dir / f"generated_{wav_file.name}"
                sample_results = self.benchmark_sample(
                    str(wav_file), text, str(output_path)
                )

                results.append(
                    {
                        "reference_file": str(wav_file),
                        "generated_file": str(output_path),
                        "text": text,
                        "metrics": sample_results,
                    }
                )

                # Update average metrics
                avg_metrics["speaker_similarity"] += sample_results[
                    "speaker_similarity"
                ]
                avg_metrics["transcription_accuracy"] += sample_results[
                    "transcription_accuracy"
                ]
                if sample_results["pesq_score"]:
                    avg_metrics["pesq_score"] += sample_results["pesq_score"]
                    valid_pesq_scores += 1

                processed_files += 1
                print(f"Processed {wav_file.name} - Text: {text[:100]}...")

            except Exception as e:
                print(f"Error processing {wav_file.name}: {str(e)}")
                continue

        if processed_files == 0:
            print("No files were successfully processed!")
            return None

        # Calculate averages
        avg_metrics["speaker_similarity"] /= processed_files
        avg_metrics["transcription_accuracy"] /= processed_files
        if valid_pesq_scores > 0:
            avg_metrics["pesq_score"] /= valid_pesq_scores

        # Save detailed results
        results_file = output_dir / "benchmark_results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "benchmark_info": {
                        "timestamp": timestamp,
                        "dataset_path": str(dataset_path),
                        "total_files": len(wav_files),
                        "processed_files": processed_files,
                        "valid_pesq_scores": valid_pesq_scores,
                    },
                    "individual_results": results,
                    "average_metrics": avg_metrics,
                },
                f,
                indent=2,
            )

        # Save summary with explanations
        summary_file = output_dir / "summary.txt"
        with open(summary_file, "w") as f:
            f.write("StyleTTS2 Benchmark Summary\n")
            f.write("=========================\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Dataset: {dataset_path}\n")
            f.write(f"Total files: {len(wav_files)}\n")
            f.write(f"Processed files: {processed_files}\n")
            f.write(f"Valid PESQ scores: {valid_pesq_scores}\n\n")

            f.write("Average Metrics:\n")
            f.write("--------------\n")
            f.write(f"Speaker Similarity: {avg_metrics['speaker_similarity']:.4f}\n")
            f.write(
                "(Range: 0-1, Higher is better. Measures how well the voice characteristics match)\n\n"
            )

            f.write(f"PESQ Score: {avg_metrics['pesq_score']:.4f}\n")
            f.write(
                "(Range: -0.5 to 4.5, Higher is better. Industry standard for audio quality)\n\n"
            )

            f.write(
                f"Transcription Accuracy: {avg_metrics['transcription_accuracy']:.4f}\n"
            )
            f.write(
                "(Range: 0-1, Higher is better. Measures how well the content/words are preserved)\n"
            )

        print(
            f"\nSuccessfully processed {processed_files} out of {len(wav_files)} files"
        )
        print(f"Results saved in: {output_dir}")
        return avg_metrics


def main():
    # Initialize the benchmark
    benchmark = VoiceBenchmark()

    # Run the benchmark and get results
    avg_metrics = benchmark.benchmark_dataset(
        dataset_path="Data_prep/raw_data/raw_audio2",
        output_dir="benchmark/results",
    )

    # Print the results with explanations
    if avg_metrics:
        print("\nBenchmark Results:")
        print("------------------")
        print(f"Speaker Similarity: {avg_metrics['speaker_similarity']:.4f}")
        print("(Range: 0-1, Higher is better)")
        print(
            "Measures how well the generated voice matches the characteristics of the original voice"
        )
        print()

        print(f"PESQ Score: {avg_metrics['pesq_score']:.4f}")
        print("(Range: -0.5 to 4.5, Higher is better)")
        print("Industry standard measure of audio quality:")
        print()

        print(f"Transcription Accuracy: {avg_metrics['transcription_accuracy']:.4f}")
        print("(Range: 0-1, Higher is better)")
        print(
            "Measures how well the generated speech preserves the original text content"
        )
    else:
        print("\nBenchmark failed: No results to display")


if __name__ == "__main__":
    main()
