import torch
import random
import yaml
import numpy as np
import torchaudio
import librosa
from nltk.tokenize import word_tokenize
import nltk
from huggingface_hub import hf_hub_download
import os
import phonemizer
from pathlib import Path
import sys
import shutil
import warnings
import logging
import re
import contextlib
from utils.emotion_utils import EMOTION_CONFIG

# Suppress all warnings
warnings.filterwarnings("ignore")
logging.getLogger("phonemizer").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# VRAM optimization - Use minimal memory
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.4)  # Limit to 40% VRAM
    torch.cuda.empty_cache()

# Add StyleTTS2 directory to Python path
styletts2_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "StyleTTS2"
)
sys.path.append(styletts2_path)


# Redirect stdout temporarily
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# Download NLTK punkt if not already downloaded
with suppress_stdout():
    nltk.download("punkt", quiet=True)

from StyleTTS2.text_utils import TextCleaner


class StyleTTS2Inference:
    def __init__(self, model_name="Amelia", repo_id=None, device=None):
        # Map model names to their HuggingFace repo IDs
        self.model_configs = {
            "Amelia": "nonoJDWAOIDAWKDA/Amelia_reviewed2_ft_StyleTTS2",
            "Eveland": "nonoJDWAOIDAWKDA/Eveland1_ft_StyleTTS2",
            "Gura": "nonoJDWAOIDAWKDA/Gura_reviewed_ft_StyleTTS2",
            "Shiori": "nonoJDWAOIDAWKDA/Shiori_reviewed_ft_StyleTTS2",
            "Wilson": "nonoJDWAOIDAWKDA/Wilson_reviewed_ft_StyleTTS2",  # Wilson's dedicated TTS model
        }

        self.model_name = model_name
        self.repo_id = repo_id if repo_id else self.model_configs[model_name]

        # Smart device selection with cuDNN fallback
        self.device = self._select_device_with_fallback()

        # Apply memory optimizations
        if self.device == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use max 80% of GPU memory

        # Create cache directory based on model name
        self.cache_dir = Path("cache/style") / model_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Add style cache
        self._style_cache = {}
        self._style_cache_dir = self.cache_dir / "style_cache"
        self._style_cache_dir.mkdir(exist_ok=True)

        # Initialize text cleaner (only once)
        with suppress_stdout():
            self.text_cleaner = TextCleaner()

        # Set seeds for reproducibility - OPTIMIZED FOR PERFORMANCE
        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = True  # Enable benchmark for faster performance
        torch.backends.cudnn.deterministic = False  # Disable deterministic for speed
        random.seed(0)
        np.random.seed(0)

        # Initialize mel spectrogram transform with optimized settings
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300
        ).to(self.device)
        self.mean, self.std = -4, 4

        # Initialize phonemizer
        self.global_phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )

        # Load all necessary components
        self._load_components()

        # Clear GPU cache after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _select_device_with_fallback(self):
        """Smart device selection with cuDNN fallback"""
        if not torch.cuda.is_available():
            return "cpu"

        try:
            # Test cuDNN availability by creating a simple convolution operation
            test_tensor = torch.randn(1, 1, 3, 3).cuda()
            conv = torch.nn.Conv2d(1, 1, 3).cuda()
            with torch.no_grad():
                _ = conv(test_tensor)
            return "cuda"
        except Exception as e:
            print(f"⚠️ cuDNN not available, using CPU: {str(e)}")
            return "cpu"

    def _download_file(self, filename):
        """Download a file from the HuggingFace repository"""
        return hf_hub_download(repo_id=self.repo_id, filename=filename)
        """Download a file from the HuggingFace repository"""
        return hf_hub_download(repo_id=self.repo_id, filename=filename)

    def _load_components(self):
        """Load all model components"""
        # Load config
        config_path = self._download_file("config.yml")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Import necessary modules
        from StyleTTS2.models import build_model, load_ASR_models, load_F0_models
        from StyleTTS2.utils import recursive_munch

        # Create cache directories for utility models
        cache_dir = Path(self.cache_dir)
        for util_dir in ["ASR", "JDC", "PLBERT"]:
            (cache_dir / "Utils" / util_dir).mkdir(parents=True, exist_ok=True)

        # Load ASR model with weights_only=True
        asr_config = self._download_file("Utils/ASR/config.yml")
        asr_model = self._download_file("Utils/ASR/epoch_00080.pth")
        self.text_aligner = load_ASR_models(asr_model, asr_config)

        # Load F0 model
        f0_model = self._download_file("Utils/JDC/bst.t7")
        self.pitch_extractor = load_F0_models(f0_model)

        # Load PLBERT
        from StyleTTS2.Utils.PLBERT.util import load_plbert

        plbert_model = self._download_file("Utils/PLBERT/step_1000000.t7")
        plbert_config = self._download_file("Utils/PLBERT/config.yml")

        # Create a temporary directory structure for PLBERT
        plbert_dir = cache_dir / "Utils" / "PLBERT"
        shutil.copy(plbert_model, plbert_dir / "step_1000000.t7")
        shutil.copy(plbert_config, plbert_dir / "config.yml")

        self.plbert = load_plbert(str(plbert_dir))

        # Build model
        model_params = recursive_munch(self.config["model_params"])
        self.model = build_model(
            model_params, self.text_aligner, self.pitch_extractor, self.plbert
        )

        # Load model weights with weights_only=True
        checkpoint = torch.load(
            self._download_file("checkpoint.pth"),
            map_location=self.device,
            weights_only=True,
        )
        params = checkpoint["net"]

        for key in self.model:
            if key in params:
                try:
                    self.model[key].load_state_dict(params[key])
                except Exception:
                    from collections import OrderedDict

                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    self.model[key].load_state_dict(new_state_dict, strict=False)

        # Move model to device and set to eval mode
        _ = [self.model[key].to(self.device) for key in self.model]
        _ = [self.model[key].eval() for key in self.model]

        # Initialize diffusion sampler
        from StyleTTS2.Modules.diffusion.sampler import (
            DiffusionSampler,
            ADPM2Sampler,
            KarrasSchedule,
        )

        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )

    def _length_to_mask(self, lengths):
        """Convert lengths to mask"""
        mask = (
            torch.arange(lengths.max())
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
            .type_as(lengths)
        )
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask

    def _preprocess(self, wave):
        """Preprocess audio waveform to mel spectrogram"""
        wave_tensor = torch.from_numpy(wave).float().to(self.device)
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def _get_cache_path(self, style_path):
        """Get the cache file path for a style"""
        style_name = Path(style_path).name
        return self._style_cache_dir / f"{style_name}.pt"

    def is_style_cached(self, path):
        """Check if a style is already cached"""
        if path in self._style_cache:
            return True
        cache_path = self._get_cache_path(path)
        return cache_path.exists()

    def get_cached_style(self, path):
        """Get a cached style vector without recomputing"""
        # Check memory cache first (fastest)
        if path in self._style_cache:
            return self._style_cache[path]

        # Load from disk cache if not in memory
        cache_path = self._get_cache_path(path)
        if cache_path.exists():
            with suppress_stdout():  # Suppress any potential PyTorch messages
                self._style_cache[path] = torch.load(
                    cache_path, map_location=self.device
                )
            return self._style_cache[path]

        raise ValueError(f"Style for {path} not found in cache")

    def compute_style(self, path):
        """Compute style vector from reference audio with caching"""
        # Check memory cache first
        if path in self._style_cache:
            return self._style_cache[path]

        # Check disk cache
        cache_path = self._get_cache_path(path)
        if cache_path.exists():
            self._style_cache[path] = torch.load(cache_path, map_location=self.device)
            return self._style_cache[path]

        # Compute new style
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = self._preprocess(audio).to(self.device)

        with torch.inference_mode():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        result = torch.cat([ref_s, ref_p], dim=1)

        # Cache in memory
        self._style_cache[path] = result

        # Cache to disk
        torch.save(result, cache_path)

        return result

    def clean_text(self, text):
        """Clean text before phonemization - minimal cleaning like official implementation"""
        text = text.strip()
        text = text.replace('"', "")  # Remove quotes only

        # Limit text length to prevent tensor size mismatch
        # StyleTTS2 typically has a max sequence length of ~512 tokens
        # Rough estimate: 1 token ≈ 1-2 characters for phonemized text
        max_chars = 800  # Conservative limit to stay well under 512 tokens

        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        return text

    def inference(
        self, text, ref_s, alpha, beta, diffusion_steps, embedding_scale, speed=1.0
    ):
        """Generate speech from text with cuDNN error handling"""
        try:
            return self._inference_internal(
                text, ref_s, alpha, beta, diffusion_steps, embedding_scale, speed
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "cudnn" in error_msg or "cuda" in error_msg:
                print(f"⚠️ CUDA/cuDNN error detected: {e}")
                print("🔄 Switching to CPU mode and retrying...")

                # Switch to CPU mode
                old_device = self.device
                self.device = "cpu"

                # Move models to CPU
                try:
                    for key in self.model:
                        self.model[key] = self.model[key].cpu()

                    # Retry inference on CPU
                    result = self._inference_internal(
                        text,
                        ref_s,
                        alpha,
                        beta,
                        diffusion_steps,
                        embedding_scale,
                        speed,
                    )
                    return result

                except Exception as cpu_error:
                    self.device = old_device  # Restore original device
                    raise cpu_error
            else:
                # Re-raise non-CUDA errors
                raise e

    def _inference_internal(
        self, text, ref_s, alpha, beta, diffusion_steps, embedding_scale, speed=1.0
    ):
        """Generate speech from text"""
        # Clean text minimally
        text = self.clean_text(text)

        if text == "":
            return np.zeros(0), 0.0

        ps = self.global_phonemizer.phonemize([text])
        if not ps or not ps[0]:
            return np.zeros(0, dtype=np.float32), 0.0

        ps = word_tokenize(ps[0])
        ps = " ".join(ps)
        tokens = self.text_cleaner(ps)
        if not tokens:  # If tokenization failed
            raise ValueError("Text tokenization failed")

        # Check token length to prevent tensor size mismatch
        max_tokens = 512  # StyleTTS2 maximum sequence length
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]

        tokens.insert(0, 0)  # Add start token
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        # Precompute input lengths and text mask
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
        text_mask = self._length_to_mask(input_lengths).to(self.device)

        # Use no_grad for inference
        with torch.inference_mode():
            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            # Create noise directly on the device
            noise = torch.randn((1, 256), device=self.device).unsqueeze(1)

            s_pred = self.sampler(
                noise=noise,
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s,
                num_steps=diffusion_steps,
            ).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            # Adjust speed
            pred_dur /= speed

            # Precompute pred_aln_trg tensor on the device
            pred_aln_trg = torch.zeros(
                input_lengths, int(pred_dur.sum().data), device=self.device
            )
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device)
            if self.config["model_params"]["decoder"]["type"] == "hifigan":
                asr_new = torch.zeros_like(
                    en, device=self.device
                )  # Create tensor on device
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = t_en @ pred_aln_trg.unsqueeze(0).to(self.device)
            if self.config["model_params"]["decoder"]["type"] == "hifigan":
                asr_new = torch.zeros_like(
                    asr, device=self.device
                )  # Create tensor on device
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        # Clean up VRAM after inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return out.squeeze().cpu().numpy()[5000:-5000]
