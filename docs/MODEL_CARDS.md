# Vbot Model Cards

Cards for every model the Vbot runtime ships. Numbers come from the
evaluation platform (`scripts/emotion_eval.py`, `scripts/llm_benchmark.py`,
`scripts/persona_judge.py`, `scripts/tts_objective_eval.py`); the standing
baselines live in [evaluation/baselines/](../evaluation/baselines/) and are
re-checked by the [Model Eval workflow](../.github/workflows/model-eval.yml).

All figures dated 2026-07-06 unless noted. "Pending" means the eval command
exists but has not been run for that model yet.

---

## Language model — `stheno` (shared across all characters)

| | |
| --- | --- |
| Source | `bluuwhale/L3-SthenoMaidBlackroot-8B-V1` (Llama-3-family 8B merge, GGUF Q4_K_M) |
| Served by | Ollama in Docker (alias `stheno`, port 11500), one instance shared by all characters |
| Character identity | Prompt layer only (`MODEL_PROMPTS` in utils/ollama_utils.py); history cleared on character switch |
| Contract | ≤30 words, first person, no emoji/parentheses/action text (enforced by prompt + runtime filter `utils/response_filter.py`) |

**Measured (10-prompt battery incl. bait prompts, ~100 tok/s on RTX 5080):**

| character | TTS-safe | brevity | persona voice* | kayfabe breaks* |
| --- | --- | --- | --- | --- |
| Amelia | 80% | 40% | 4.80 | 0% |
| Eveland | 80% | 30% | 4.10 | 20% |
| Gura | 70% | 30% | 5.00 | 0% |
| Shiori | 80% | 40% | 4.70 | 20% |
| Wilson | 80% | 40% | 3.80 | 20% |

\* Persona/kayfabe scored 1–5 by an independent LLM judge (`mistral:latest`,
judge prompt v2 — v1 was discarded after failing a discrimination check;
see the calibration story in the private tracker). Judge is report-only, not
a gate: a 7B judge is noisy and smears dimensions.

**Known limitations:** brevity contract is the weakest (avg 40–48 words vs
the 30-word instruction; runtime truncates TTS input at ~100 words).
Identity probes break kayfabe for Eveland/Shiori/Wilson — confirmed by both
the heuristic detector and the judge. Roleplay-bait induces action text in
3/5 characters; the runtime filter strips it before speech.

---

## Emotion classifier — `SamLowe/roberta-base-go_emotions`

| | |
| --- | --- |
| Task | Classify LLM responses into 28 GoEmotions labels → mapped to 5 voice-style buckets (neutral/happy/sad/angry/surprised) |
| Runtime | CPU, one shared process-wide pipeline (consolidated 2026-07-06: was ~10 duplicate instances, −3.0 GB RAM) |
| Confidence threshold | 0.15 (calibrated by threshold sweep; the historical 0.3 was past the knee on both eval slices) |
| Eval datasets | Frozen in [evaluation/emotion/](../evaluation/emotion/): 741-item GoEmotions test slice + 80-item hand-curated domain slice |

**Measured (runtime-faithful: label → threshold → bucket):**

| dataset | accuracy | macro-F1 | weakest bucket |
| --- | --- | --- | --- |
| GoEmotions slice | 64.5% | 0.640 | surprised (recall 0.35) |
| Domain slice | 77.5% | 0.787 | angry / neutral (F1 0.64) |

**Known limitations:** trained on Reddit comments, not conversational
VTuber dialogue. Surprise/realization is systematically confused with
neutral/curiosity — a classifier weakness, not a threshold artifact. The
neutral bucket absorbs errors by design (6 labels incl. curiosity map to
it). Swap candidates should be evaluated with
`scripts/emotion_eval.py --model <hf-model> --baseline evaluation/baselines/emotion_eval_baseline.json`.

---

## Character voices — fine-tuned StyleTTS2

Common to all five: base architecture StyleTTS2 (LibriTTS multispeaker
checkpoint), fine-tuned per character; 24 kHz output; emotion delivery via
per-emotion reference styles in `asset/ref_sound/<Character>/`; inference
params in `utils/emotion_utils.MODEL_PARAMS` (currently uniform
alpha 0.3 / beta 0.7 — per-character tuning is future work).

### Amelia — `nonoJDWAOIDAWKDA/Amelia_reviewed2_ft_StyleTTS2`

| metric | value | source |
| --- | --- | --- |
| Speaker similarity (vs reference recordings) | **0.819** | tts_objective_eval, 10-sentence battery |
| WER (Whisper base transcription) | **0.058** | same artifact (absolute value inflated ~0.05 by ASR number formatting; relative comparisons unaffected) |
| Synthesis speed | RTF 0.14–0.40 on RTX 5080 (~0.85s fixed cost + ~0.02s/word) | latency profile 2026-07-06 |

### Eveland — `nonoJDWAOIDAWKDA/Eveland1_ft_StyleTTS2`

| metric | value |
| --- | --- |
| Speaker similarity | **0.873** |
| WER | **0.027** |

### Gura — `nonoJDWAOIDAWKDA/Gura_reviewed_ft_StyleTTS2`

| metric | value |
| --- | --- |
| Speaker similarity | **0.871** |
| WER | **0.057** |

### Shiori — `nonoJDWAOIDAWKDA/Shiori_reviewed_ft_StyleTTS2`

| metric | value |
| --- | --- |
| Speaker similarity | **0.863** |
| WER | **0.037** |

### Wilson — `nonoJDWAOIDAWKDA/Wilson_reviewed_ft_StyleTTS2`

| metric | value |
| --- | --- |
| Speaker similarity | **0.856** |
| WER | **0.065** |

All five voices have standing objective baselines in
`evaluation/baselines/tts_objective_<Character>_baseline.json`
(2026-07-06, 10-sentence battery v1, Whisper base). Notably, Amelia — the
oldest fine-tune — has the *lowest* speaker similarity (0.819) of the
five; the newer "reviewed" fine-tunes cluster at 0.856–0.873. Worth
revisiting Amelia's voice with the current fine-tuning recipe.

**Human evaluation:** emotion recognizability and naturalness are collected
via the [TTS eval form](../new_tts_eval_form/) (schema v1 artifacts);
promotion of any candidate voice requires
`new_tts_eval_form/promotion_gate.py` to pass on human + objective metrics
combined.

---

## Avatar animation — THA4 (for completeness)

Not an in-repo trained model: character models are distilled THA4 poser
networks (`asset/model/<Character>/`, face/body morpher state dicts).
Measured ~127 FPS at 512×512 fp32 on RTX 5080, 0.47 GB peak VRAM (app
targets ~30 FPS). See `scripts/gpu_smoke_test.py` stage 2.
