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
import shutil
from pystoi import stoi
from mir_eval.separation import bss_eval_sources
import re
import gc

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))
from utils.inference_styleTTS2 import StyleTTS2Inference

# List of models to test (repo IDs)
MODELS_TO_TEST = [
    "nonoJDWAOIDAWKDA/new_ft_StyleTTS2",
    "nonoJDWAOIDAWKDA/new2_ft_StyleTTS2",
    "nonoJDWAOIDAWKDA/Amelia1_ft_StyleTTS2",
    "nonoJDWAOIDAWKDA/Amelia2_ft_StyleTTS2",
    "nonoJDWAOIDAWKDA/Amelia3_ft_StyleTTS2",
    "nonoJDWAOIDAWKDA/Amelia4_ft_StyleTTS2",
    "nonoJDWAOIDAWKDA/Amelia5_ft_StyleTTS2",
    "nonoJDWAOIDAWKDA/Amelia6_ft_StyleTTS2",
    "nonoJDWAOIDAWKDA/Amelia7_ft_StyleTTS2",
    "nonoJDWAOIDAWKDA/Amelia8_ft_StyleTTS2",
    "nonoJDWAOIDAWKDA/Amelia9_ft_StyleTTS2",
    "nonoJDWAOIDAWKDA/Amelia10_ft_StyleTTS2",
]


class VoiceBenchmark:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.tts = (
            StyleTTS2Inference(repo_id=model_path)
            if model_path
            else StyleTTS2Inference()
        )
        self.voice_encoder = VoiceEncoder()

        # Initialize transcription model using transformers pipeline
        from transformers import pipeline
        from transformers.utils import is_flash_attn_2_available

        print("Initializing transcription model...")
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model="distil-whisper/large-v2",  # Using distilled model for better speed
            torch_dtype=torch.float16,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_kwargs=(
                {"attn_implementation": "flash_attention_2"}
                if is_flash_attn_2_available()
                else {"attn_implementation": "sdpa"}
            ),
        )
        print("Transcription model initialized")

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
        """Transcribe audio using distil-whisper"""
        try:
            # Load audio file
            audio = librosa.load(audio_path, sr=16000)[0]

            # Transcribe audio with optimized settings
            result = self.transcriber(
                {"array": audio, "sampling_rate": 16000},
                batch_size=1,  # Single batch since we're processing one file at a time
                generate_kwargs={"task": "transcribe", "language": "en"},
            )

            # Get transcription text and clean it
            text = result["text"].strip()
            text = re.sub(
                r"\s+", " ", text
            )  # Replace multiple spaces with single space
            text = text.strip().lower()

            return text
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            return None

    def compute_transcription_accuracy(self, original_text, generated_text):
        """Compute word-level accuracy between original and generated transcriptions"""
        original_words = set(original_text.split())
        generated_words = set(generated_text.split())

        if not original_words:
            return 0.0

        common_words = original_words.intersection(generated_words)
        return len(common_words) / len(original_words)

    def compute_stoi_score(self, reference_path, generated_path):
        """Compute STOI score between reference and generated audio"""
        ref_wav, sr = librosa.load(reference_path, sr=24000)
        gen_wav, _ = librosa.load(generated_path, sr=24000)

        # Resample to 16000Hz for STOI calculation
        ref_wav = librosa.resample(ref_wav, orig_sr=24000, target_sr=16000)
        gen_wav = librosa.resample(gen_wav, orig_sr=24000, target_sr=16000)

        # Ensure both audio files have the same length
        min_len = min(len(ref_wav), len(gen_wav))
        ref_wav = ref_wav[:min_len]
        gen_wav = gen_wav[:min_len]

        return stoi(ref_wav, gen_wav, 16000)

    def compute_sdr_score(self, reference_path, generated_path):
        """Compute SDR score between reference and generated audio"""
        ref_wav, sr = librosa.load(reference_path, sr=24000)
        gen_wav, _ = librosa.load(generated_path, sr=24000)

        # Ensure both audio files have the same length
        min_len = min(len(ref_wav), len(gen_wav))
        ref_wav = ref_wav[:min_len]
        gen_wav = gen_wav[:min_len]

        sdr, _, _, _ = bss_eval_sources(ref_wav, gen_wav)
        return sdr[0]

    def compute_mel_cepstral_distortion(self, reference_path, generated_path):
        """Compute Mel Cepstral Distortion between reference and generated speech"""
        try:
            # Load audio files
            ref_wav, sr = librosa.load(reference_path, sr=24000)
            gen_wav, _ = librosa.load(generated_path, sr=24000)

            # Ensure same length
            min_len = min(len(ref_wav), len(gen_wav))
            ref_wav = ref_wav[:min_len]
            gen_wav = gen_wav[:min_len]

            # Extract MFCCs
            ref_mfcc = librosa.feature.mfcc(y=ref_wav, sr=sr, n_mfcc=13)
            gen_mfcc = librosa.feature.mfcc(y=gen_wav, sr=sr, n_mfcc=13)

            # Compute MCD
            mcd = np.mean(np.sqrt(np.sum((ref_mfcc - gen_mfcc) ** 2, axis=0)))
            return float(mcd)
        except Exception as e:
            print(f"Error computing MCD: {str(e)}")
            return None

    def compute_prosody_similarity(self, reference_path, generated_path):
        """Compare prosodic features like pitch contour and rhythm"""
        try:
            # Load audio files
            ref_wav, sr = librosa.load(reference_path, sr=24000)
            gen_wav, _ = librosa.load(generated_path, sr=24000)

            # Ensure same length
            min_len = min(len(ref_wav), len(gen_wav))
            ref_wav = ref_wav[:min_len]
            gen_wav = gen_wav[:min_len]

            # Extract pitch
            ref_pitch, _ = librosa.piptrack(y=ref_wav, sr=sr)
            gen_pitch, _ = librosa.piptrack(y=gen_wav, sr=sr)

            # Compare pitch contours
            ref_pitch_mean = np.mean(ref_pitch, axis=1)
            gen_pitch_mean = np.mean(gen_pitch, axis=1)

            # Ensure non-zero variance
            if np.var(ref_pitch_mean) == 0 or np.var(gen_pitch_mean) == 0:
                return 0.0

            # Calculate correlation
            pitch_sim = np.corrcoef(ref_pitch_mean, gen_pitch_mean)[0, 1]

            # Handle NaN values
            if np.isnan(pitch_sim):
                return 0.0

            return float(pitch_sim)
        except Exception as e:
            print(f"Error computing prosody similarity: {str(e)}")
            return None

    def compute_spectral_convergence(self, reference_path, generated_path):
        """Compute spectral convergence between reference and generated speech"""
        try:
            # Load audio files
            ref_wav, sr = librosa.load(reference_path, sr=24000)
            gen_wav, _ = librosa.load(generated_path, sr=24000)

            # Ensure same length
            min_len = min(len(ref_wav), len(gen_wav))
            ref_wav = ref_wav[:min_len]
            gen_wav = gen_wav[:min_len]

            # Compute spectrograms
            ref_spec = np.abs(librosa.stft(ref_wav))
            gen_spec = np.abs(librosa.stft(gen_wav))

            # Compute convergence
            spec_conv = np.linalg.norm(ref_spec - gen_spec) / np.linalg.norm(ref_spec)
            return float(spec_conv)
        except Exception as e:
            print(f"Error computing spectral convergence: {str(e)}")
            return None

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

        # Compute STOI score
        stoi_score = self.compute_stoi_score(reference_path, output_path)

        # Compute SDR score
        sdr_score = self.compute_sdr_score(reference_path, output_path)

        # Add new metrics to the results
        sample_results = {
            "speaker_similarity": float(speaker_similarity),
            "pesq_score": float(pesq_score) if pesq_score else None,
            "transcription_accuracy": float(transcription_accuracy),
            "stoi_score": float(stoi_score),
            "sdr_score": float(sdr_score),
            "original_transcription": original_transcription,
            "generated_transcription": generated_transcription,
            "mel_cepstral_distortion": self.compute_mel_cepstral_distortion(
                reference_path, output_path
            ),
            "prosody_similarity": self.compute_prosody_similarity(
                reference_path, output_path
            ),
            "spectral_convergence": self.compute_spectral_convergence(
                reference_path, output_path
            ),
        }

        return sample_results

    def explain_metrics(self):
        """Provide detailed explanations for each metric used in the benchmark"""
        explanations = {
            "speaker_similarity": (
                "Speaker Similarity: Measures how well the generated voice matches the "
                "characteristics of the original voice. Calculated using cosine similarity "
                "between speaker embeddings. Range: 0-1, Higher is better. A higher score "
                "indicates that the generated voice closely resembles the original speaker's voice."
            ),
            "pesq_score": (
                "PESQ Score: Industry standard measure of audio quality. Compares reference "
                "and generated audio to evaluate naturalness and clarity. Range: -0.5 to 4.5, "
                "Higher is better. A higher PESQ score suggests that the generated audio is "
                "perceived as more natural and clear."
            ),
            "transcription_accuracy": (
                "Transcription Accuracy: Measures how well the generated speech preserves "
                "the original text content. Calculated as the ratio of common words between "
                "original and generated transcriptions. Range: 0-1, Higher is better. This metric "
                "is sensitive to transcription errors, including spelling mistakes and word order. "
                "If a word is off by even one letter, it is considered different."
            ),
            "stoi_score": (
                "STOI Score: Measures the intelligibility of the generated speech. Compares "
                "short-time segments of reference and generated audio. Range: 0-1, Higher is better. "
                "A higher STOI score means the generated speech is clearer and more understandable."
            ),
            "sdr_score": (
                "SDR Score: Measures the distortion in the generated audio compared to the reference. "
                "Calculated using the BSS Eval toolkit. Higher is better. A higher SDR score indicates "
                "less distortion and better fidelity to the original audio."
            ),
            "mel_cepstral_distortion": (
                "Mel Cepstral Distortion (MCD): Measures the difference in spectral "
                "features between reference and generated speech. Lower is better. "
                "This metric is particularly good at capturing voice timbre differences."
            ),
            "prosody_similarity": (
                "Prosody Similarity: Measures how well the generated speech matches the "
                "reference speech in terms of pitch patterns and rhythm. Range: -1 to 1, "
                "where 1 indicates perfect correlation. This helps evaluate the naturalness "
                "of speech patterns."
            ),
            "spectral_convergence": (
                "Spectral Convergence: Measures how well the spectral content of the "
                "generated speech matches the reference. Lower values indicate better "
                "matching of frequency components. This helps evaluate overall audio quality "
                "and voice characteristics."
            ),
        }
        return explanations

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
            model_name/
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
        model_name = self.model_path.split("/")[-1] if self.model_path else "default"
        output_dir = base_output_dir / model_name / timestamp
        samples_dir = output_dir / "generated_samples"

        # Create directory structure
        output_dir.mkdir(parents=True, exist_ok=True)
        samples_dir.mkdir(exist_ok=True)

        dataset_path = Path(dataset_path)
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
            "stoi_score": 0,
            "sdr_score": 0,
            "mel_cepstral_distortion": 0,
            "prosody_similarity": 0,
            "spectral_convergence": 0,
        }
        valid_pesq_scores = 0
        processed_files = 0

        for wav_file in tqdm(wav_files, desc=f"Benchmarking samples for {model_name}"):
            try:
                # Transcribe the original audio
                print(f"\nTranscribing {wav_file.name}...")
                text = self.transcribe_audio(str(wav_file))

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
                avg_metrics["stoi_score"] += sample_results["stoi_score"]
                avg_metrics["sdr_score"] += sample_results["sdr_score"]
                if sample_results["pesq_score"]:
                    avg_metrics["pesq_score"] += sample_results["pesq_score"]
                    valid_pesq_scores += 1

                # Update averages for new metrics
                for metric in avg_metrics:
                    if (
                        metric != "pesq_score"
                    ):  # PESQ is handled separately due to potential None values
                        avg_metrics[metric] += sample_results[metric]

                processed_files += 1
                print(f"Processed {wav_file.name} - Text: {text[:100]}...")

            except Exception as e:
                print(f"Error processing {wav_file.name}: {str(e)}")
                continue

        if processed_files == 0:
            print("No files were successfully processed!")
            return None

        # Calculate averages
        for metric in avg_metrics:
            if (
                metric != "pesq_score"
            ):  # PESQ is handled separately due to potential None values
                avg_metrics[metric] /= processed_files

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
                        "model_path": self.model_path,
                    },
                    "individual_results": results,
                    "average_metrics": avg_metrics,
                },
                f,
                indent=2,
            )

        # Save summary with explanations
        explanations = self.explain_metrics()
        summary_file = output_dir / "summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"StyleTTS2 Benchmark Summary for {model_name}\n")
            f.write("=" * (35 + len(model_name)) + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Dataset: {dataset_path}\n")
            f.write(f"Total files: {len(wav_files)}\n")
            f.write(f"Processed files: {processed_files}\n")
            f.write(f"Valid PESQ scores: {valid_pesq_scores}\n\n")

            f.write("Average Metrics:\n")
            f.write("----------------\n")
            f.write(
                f"Speaker Similarity: {avg_metrics['speaker_similarity']:.4f} (Range: 0-1, Higher is better)\n"
            )
            f.write(explanations["speaker_similarity"] + "\n\n")

            f.write(
                f"PESQ Score: {avg_metrics['pesq_score']:.4f} (Range: -0.5 to 4.5, Higher is better)\n"
            )
            f.write(explanations["pesq_score"] + "\n\n")

            f.write(
                f"Transcription Accuracy: {avg_metrics['transcription_accuracy']:.4f} (Range: 0-1, Higher is better)\n"
            )
            f.write(explanations["transcription_accuracy"] + "\n\n")

            f.write(
                f"STOI Score: {avg_metrics['stoi_score']:.4f} (Range: 0-1, Higher is better)\n"
            )
            f.write(explanations["stoi_score"] + "\n\n")

            f.write(f"SDR Score: {avg_metrics['sdr_score']:.4f} (Higher is better)\n")
            f.write(explanations["sdr_score"] + "\n\n")

            f.write(
                f"Mel Cepstral Distortion: {avg_metrics['mel_cepstral_distortion']:.4f} (Lower is better)\n"
            )
            f.write(explanations["mel_cepstral_distortion"] + "\n\n")

            f.write(
                f"Prosody Similarity: {avg_metrics['prosody_similarity']:.4f} (Range: -1 to 1, Higher is better)\n"
            )
            f.write(explanations["prosody_similarity"] + "\n\n")

            f.write(
                f"Spectral Convergence: {avg_metrics['spectral_convergence']:.4f} (Lower is better)\n"
            )
            f.write(explanations["spectral_convergence"] + "\n\n")

        print(
            f"\nSuccessfully processed {processed_files} out of {len(wav_files)} files"
        )
        print(f"Results saved in: {output_dir}")
        return avg_metrics

    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, "transcriber"):
                del self.transcriber
            if hasattr(self, "tts"):
                del self.tts
            if hasattr(self, "voice_encoder"):
                del self.voice_encoder

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Error during cleanup: {str(e)}")


def clear_style_cache():
    """Clear all style cache directories"""
    cache_dir = Path("cache/style")
    if cache_dir.exists():
        print("Clearing style cache...")
        shutil.rmtree(cache_dir)
        print("Style cache cleared.")
    else:
        print("No style cache found.")


def cleanup_model(benchmark):
    """Clean up model resources"""
    try:
        if benchmark is not None:
            benchmark.cleanup()
            del benchmark

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Error during cleanup: {str(e)}")


def main():
    try:
        # Clear style cache before starting
        clear_style_cache()

        # Base output directory for all results
        base_output_dir = Path("benchmark/results")
        base_output_dir.mkdir(parents=True, exist_ok=True)

        # Store all models' results for comparison
        all_results = {}
        failed_models = []
        total_models = len(MODELS_TO_TEST)

        print(f"\nStarting benchmark of {total_models} models")
        print("=" * 80)

        # Create a progress tracking file
        progress_file = base_output_dir / "benchmark_progress.txt"
        with open(progress_file, "w", encoding="utf-8") as f:
            f.write(
                f"Starting benchmark at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Total models to test: {total_models}\n\n")

        # Test each model
        for idx, model_path in enumerate(MODELS_TO_TEST, 1):
            model_error = None
            benchmark = None
            try:
                print(f"\nTesting model {idx}/{total_models}: {model_path}")
                print("=" * 80)

                # Update progress file
                with open(progress_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"\nStarting model {idx}/{total_models}: {model_path} at {datetime.now().strftime('%H:%M:%S')}\n"
                    )

                # Initialize benchmark for this model
                print("Initializing benchmark...")
                try:
                    benchmark = VoiceBenchmark(model_path)
                except Exception as init_error:
                    print(f"Error initializing model: {str(init_error)}")
                    raise init_error

                # Run the benchmark
                print("Running benchmark...")
                avg_metrics = benchmark.benchmark_dataset(
                    dataset_path="Data_prep/raw_data/raw_audio2",
                    output_dir=str(base_output_dir),
                )

                if avg_metrics:
                    all_results[model_path] = avg_metrics
                    print(f"\nResults for {model_path}:")
                    print("-" * 40)
                    for metric, value in avg_metrics.items():
                        print(f"{metric.capitalize()}: {value:.4f}")

                    # Update progress file with success
                    with open(progress_file, "a", encoding="utf-8") as f:
                        f.write(
                            f"[SUCCESS] Completed at {datetime.now().strftime('%H:%M:%S')}\n"
                        )
                        for metric, value in avg_metrics.items():
                            f.write(f"{metric.capitalize()}: {value:.4f}\n")
                else:
                    print(f"\nBenchmark failed for {model_path}")
                    failed_models.append(model_path)
                    model_error = "No results generated"
                    # Update progress file with failure
                    with open(progress_file, "a", encoding="utf-8") as f:
                        f.write(f"[FAILED] No results generated\n")

            except Exception as e:
                print(f"Error testing model {model_path}: {str(e)}")
                failed_models.append(model_path)
                model_error = str(e)
                # Update progress file with error
                try:
                    with open(progress_file, "a", encoding="utf-8") as f:
                        f.write(f"[ERROR] {str(e)}\n")
                except:
                    pass

            finally:
                # Clean up resources
                if benchmark is not None:
                    print("Cleaning up resources...")
                    cleanup_model(benchmark)
                    del benchmark

                # Save intermediate comparative results after each model
                try:
                    if all_results:
                        comparative_file = base_output_dir / "comparative_results.json"
                        with open(comparative_file, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "timestamp": datetime.now().strftime(
                                        "%Y-%m-%d_%H-%M-%S"
                                    ),
                                    "models_tested": MODELS_TO_TEST,
                                    "completed_models": list(all_results.keys()),
                                    "failed_models": failed_models,
                                    "results": all_results,
                                    "last_error": model_error,
                                },
                                f,
                                indent=2,
                            )
                        print(
                            f"\nIntermediate comparative results saved to: {comparative_file}"
                        )
                except Exception as e:
                    print(f"Warning: Failed to save comparative results: {str(e)}")

                # Add a delay between models to ensure cleanup is complete
                import time

                time.sleep(5)

        # Final summary
        print("\nBenchmark Complete!")
        print("=" * 80)
        print(f"Successfully tested: {len(all_results)}/{total_models} models")
        if failed_models:
            print("\nFailed models:")
            for model in failed_models:
                print(f"- {model}")

        # Update progress file with final summary
        try:
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write(
                    f"\nBenchmark completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(
                    f"Successfully tested: {len(all_results)}/{total_models} models\n"
                )
                if failed_models:
                    f.write("\nFailed models:\n")
                    for model in failed_models:
                        f.write(f"- {model}\n")

            if all_results:
                print(f"\nFinal comparative results saved to: {comparative_file}")
            else:
                print("\nNo results to compare - all benchmarks failed")
        except Exception as e:
            print(f"Warning: Failed to write final summary: {str(e)}")

    except Exception as e:
        print(f"Critical error in main benchmark loop: {str(e)}")
        raise


if __name__ == "__main__":
    main()
