"""Staged GPU smoke test for the Vbot inference stack.

Verifies the GPU environment end-to-end after a hardware or PyTorch change:
CUDA math, THA4 poser (with FPS), emotion classifier, StyleTTS2 synthesis.
Each stage exercises a real app code path; stages tolerate failure and
report a summary so one broken subsystem doesn't hide the others.

Usage:
    python scripts/gpu_smoke_test.py
"""

import os
import sys
import time
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("PROJECT_ROOT", PROJECT_ROOT)
os.environ.setdefault("VOICE_TYPE", "Amelia")

RESULTS = []


def stage(name):
    def decorator(func):
        def wrapper():
            print(f"\n{'=' * 60}\nSTAGE: {name}\n{'=' * 60}")
            start = time.time()
            try:
                func()
                RESULTS.append((name, True, f"{time.time() - start:.1f}s"))
                print(f"[PASS] {name}")
            except Exception as exc:
                traceback.print_exc()
                RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
                print(f"[FAIL] {name}")
        return wrapper
    return decorator


@stage("1. CUDA sanity (device, capability, math correctness)")
def stage_cuda():
    import torch

    print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
    assert torch.cuda.is_available(), "CUDA not available"
    props = torch.cuda.get_device_properties(0)
    print("device:", props.name, "| capability:", f"sm_{props.major}{props.minor}")
    print("VRAM:", f"{props.total_memory / 1024**3:.1f} GB")

    # fp32 and fp16 matmul on GPU must match CPU reference.
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    ref = a @ b
    gpu32 = (a.cuda() @ b.cuda()).cpu()
    assert torch.allclose(ref, gpu32, atol=1e-3), "fp32 GPU matmul mismatch"
    gpu16 = (a.cuda().half() @ b.cuda().half()).float().cpu()
    assert torch.allclose(ref, gpu16, atol=1.0), "fp16 GPU matmul mismatch"
    torch.cuda.synchronize()
    print("fp32/fp16 matmul: correct")


@stage("2. THA4 poser: load Amelia + pose forward + FPS")
def stage_tha4():
    import torch
    from tha4.charmodel.character_model import CharacterModel

    device = torch.device("cuda")
    yaml_path = os.path.join(
        PROJECT_ROOT, "asset", "model", "Amelia", "character_model", "character_model.yaml"
    )
    model = CharacterModel.load(yaml_path)
    image = model.get_character_image(device)
    poser = model.get_poser(device)
    print("poser dtype:", poser.get_dtype(), "| params:", poser.get_num_parameters())

    # Mirror utils/avatar.py: fp16 pose tensor, poser.pose(image, pose).
    pose = torch.zeros(
        (1, poser.get_num_parameters()), device=device, dtype=torch.float16
    )
    output = poser.pose(image, pose)
    torch.cuda.synchronize()
    print("pose output:", tuple(output.shape), output.dtype)

    # FPS over 60 frames with a moving mouth parameter (speaking animation).
    import math

    frames = 60
    start = time.time()
    for i in range(frames):
        pose[0, 0] = abs(math.sin(i * 0.3)) * 0.5
        poser.pose(image, pose)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"THA4 pose FPS: {frames / elapsed:.1f} (app targets ~30)")
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


@stage("3. Emotion classifier (RoBERTa pipeline)")
def stage_emotion():
    from utils.emotion_utils import EmotionHandler

    handler = EmotionHandler(model_name="Amelia")
    emotion = handler.classify_emotion("I am so happy to see you again!")
    print("emotion:", emotion, "| confidence:", f"{handler.get_last_confidence():.3f}")
    # Mirror InferenceHandler.process_text: config is keyed by classifier label.
    emotion_params = handler.emotion_config[emotion]
    print("style file for emotion:", emotion_params["file"]["Amelia"])


@stage("4. StyleTTS2 synthesis (Amelia, GPU)")
def stage_tts():
    import numpy as np
    from utils.emotion_utils import DIFFUSION_STEPS, create_emotion_config
    from utils.inference_styleTTS2 import StyleTTS2Inference

    tts = StyleTTS2Inference(model_name="Amelia")
    print("TTS device:", tts.device)

    # Mirror InferenceHandler.process_text path resolution and parameters.
    emotion_params = create_emotion_config("Amelia")["neutral"]
    style_path = os.path.join(
        PROJECT_ROOT, "asset", "ref_sound", emotion_params["file"]["Amelia"]
    )
    print("style reference:", style_path)
    ref_s = tts.compute_style(style_path)

    result = tts.inference(
        text="Hello! The new graphics card is working.",
        ref_s=ref_s,
        alpha=emotion_params["alpha"],
        beta=emotion_params["beta"],
        diffusion_steps=DIFFUSION_STEPS,
        embedding_scale=emotion_params["embedding_scale"],
    )
    wav = result[0] if isinstance(result, tuple) else result
    wav = np.asarray(wav)
    print("waveform:", wav.shape, wav.dtype, "| max amp:", float(abs(wav).max()))
    assert wav.size > 1000, "waveform suspiciously short"

    import soundfile as sf

    out = os.path.join(PROJECT_ROOT, "asset", "outputs", "gpu_smoke_amelia.wav")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sf.write(out, wav, 24000)
    print("wrote:", out)


if __name__ == "__main__":
    stage_cuda()
    stage_tha4()
    stage_emotion()
    stage_tts()

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for name, passed, detail in RESULTS:
        print(f"[{'PASS' if passed else 'FAIL'}] {name} ({detail})")
    sys.exit(0 if all(passed for _, passed, _ in RESULTS) else 1)
