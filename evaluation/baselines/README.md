# Evaluation Baseline Registry

Committed, version-controlled baseline artifacts that the evaluation gates
compare against. Replacing a file here **is** a model promotion: the git
history of this directory is the audit trail of what shipped, when, and
with what measured quality.

| File | Produced by | Gated by | Baseline as of |
| --- | --- | --- | --- |
| `emotion_eval_baseline.json` | `scripts/emotion_eval.py` | `scripts/emotion_eval.py --baseline` (macro-F1 + per-bucket F1, no regression) | 2026-07-06, threshold 0.15 calibration |
| `tts_objective_<Character>_baseline.json` (all 5 characters) | `scripts/tts_objective_eval.py --character <Character>` | `scripts/tts_objective_eval.py --baseline` and `new_tts_eval_form/promotion_gate.py` (speaker similarity ≥, WER ≤) | 2026-07-06, production voices |
| `persona_judged_reference.json` | `scripts/persona_judge.py` (judge prompt v2) | report-only — judge scores are compared in the eval summary but do not block (7B judge noise; see tracker session 6) | 2026-07-06, stheno + production prompts |

## Promotion procedure

1. Run the relevant eval command for the candidate (e.g.
   `python scripts/tts_objective_eval.py --character Amelia --repo-id <candidate> --label candidate`).
2. Run the gate against the current baseline in this directory. It must
   exit 0.
3. For TTS voices, also collect human ratings via `new_tts_eval_form` and
   run `promotion_gate.py` with both the human results and the objective
   artifacts.
4. Copy the candidate artifact over the baseline file here, commit with a
   message explaining the promotion, and ship the model change in the same
   PR.

Never edit these JSON files by hand — only replace them with artifacts
produced by the eval commands.
