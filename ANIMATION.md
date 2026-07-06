# Vbot Animation Implementation

This document explains Vbot's animation layer: how the app turns emotion labels, speaking state, and character assets into a live THA4 avatar inside the desktop UI.

For the procedural pose math behind blinking, gaze motion, emotion curves, and speaking-mouth movement, see [ANIMATION_MATH.md](ANIMATION_MATH.md). For the deeper THA4 training and GPU optimization analysis, see [THA4_OPTIMIZATION.md](THA4_OPTIMIZATION.md).

THA4 provides the neural avatar framework. Vbot's implementation is the runtime control layer around it:

- selecting the correct character asset
- loading the THA4 poser
- mapping emotion states into THA4 pose parameters
- generating idle motion
- adding speaking mouth motion during TTS playback
- rendering transparent avatar frames into a wxPython panel
- handling performance and fallback behavior

## Runtime Components

| Component | File | Role |
| --- | --- | --- |
| Avatar controller | `utils/avatar.py` | Loads character assets, owns the wx panel, updates pose frames |
| Animation states | `tha4/app/animations/*.py` | Idle, happy, sad, angry, surprise motion curves |
| THA4 poser | `tha4/` | Neural poser/morpher framework |
| Audio trigger | `utils/ollama_utils.py` | Starts/stops speaking animation around TTS playback |
| Emotion source | `utils/emotion_utils.py` | Maps text emotions into runtime categories |

## Character Asset Selection

`AnimatedCharacter` chooses its character from the `VOICE_TYPE` environment variable:

```text
VOICE_TYPE=Amelia
VOICE_TYPE=Gura
VOICE_TYPE=Eveland
VOICE_TYPE=Shiori
VOICE_TYPE=Wilson
```

That value decides:

- avatar model folder: `asset/model/<character>/`
- character model config: `character_model/character_model.yaml`
- background color: `bg_color.txt`
- default background image
- THA4 character assets and poser files

The seamless interface updates `VOICE_TYPE` before recreating/selecting the avatar during a character switch. That keeps the visible avatar aligned with the active TTS and prompt stack.

## Render Loop

The animation loop is driven by a wx timer:

```text
wx.Timer -> update_animation() every ~33 ms -> ~30 FPS
```

Each frame:

1. compute `delta_time`
2. create a zeroed THA4 pose vector with `poser.num_parameters`
3. choose the current animation state from `target_emotion`
4. ask the animation state for procedural values
5. apply explicit emotion pose overrides
6. apply common motion values such as breathing, head, body, eyes, and blink
7. if speaking, overlay procedural mouth motion
8. convert the pose list to a `torch.float16` tensor on the selected device
9. call `poser.pose(cached_character_image, pose_tensor)`
10. move the result back to CPU
11. convert THA4 output to a NumPy RGBA image
12. convert it into a wx bitmap
13. repaint the panel

The renderer uses a cached base character image and can fall back to static rendering when neural posing assets are unavailable.

## Emotion State Mapping

The emotion classifier produces fine-grained GoEmotions labels. `AnimatedCharacter.set_emotion()` maps those into a smaller animation vocabulary:

| Runtime state | Example source labels |
| --- | --- |
| happy | admiration, amusement, approval, excitement, gratitude, joy, love, optimism, pride |
| sad | disappointment, grief, remorse, sadness |
| angry | anger, annoyance, disapproval, disgust |
| surprise | realization, surprise |
| keep current | neutral, confusion, caring, curiosity, desire, relief |

The important behavior is that neutral labels do not automatically reset the face. If the avatar is happy, a weak/neutral classifier output keeps the current expression instead of snapping back to neutral. That makes the animation feel more stable during normal conversation.

## Pose Parameter Strategy

THA4 exposes grouped pose parameters. Vbot resolves parameter indices by category and optional group name:

- breathing
- face rotation
- body rotation
- iris rotation
- iris morph
- eye parameters
- eyebrow parameters
- mouth parameters

The pose update code writes only parameters that exist for the current poser. Missing parameters return `-1` and are skipped, which lets the same control logic survive small differences between character models.

## Idle Animation

The idle state is not a static neutral pose. `IdleAnimation` adds small motion so the avatar stays alive while listening.

It combines:

- breathing cycle
- random blinking every few seconds
- weighted look targets
- smooth eye movement
- head follow with lag
- subtle body sway
- micro-movement noise
- gentle head tilt

The look-target system is one of the more important details. It chooses from weighted "interest points" around the screen, interpolates to the next target, and lets the head follow after a delay. Eyes move first, then the head catches up. This is a simple way to make the avatar feel attentive without needing camera tracking.

## Emotion Animation States

Each emotion state owns a procedural motion profile.

### Happy

Happy animation uses:

- bouncy head/body motion
- raised mouth corners
- more open mouth shape
- happy eyebrows
- relaxed/happy eyes
- brighter iris movement
- faster breathing/micro-motion

The goal is a lively expression with more vertical movement and open facial parameters.

### Sad

Sad animation uses:

- downward head tilt
- slower movement cycles
- drooped eyes
- troubled eyebrows
- lowered mouth corners
- reduced iris motion
- slower breathing
- subtle mouth quiver

The face and body bias downward, and the timing is intentionally slower.

### Angry

Angry animation uses:

- sharper head micro-motion
- narrowed/unimpressed eyes
- angry eyebrows
- tense mouth delta
- lowered mouth corners
- more tension in body rotation
- faster breathing cycle

The profile is more rigid and sharp than happy or idle.

### Surprise

Surprise animation uses:

- wider eyes
- raised eyebrows
- rounded mouth
- small bounce
- reduced but noticeable head motion
- iris size pulsing
- centered gaze

The goal is a wide-eyed reaction without making the head motion too chaotic.

## Speaking Animation

The stable runtime path currently uses procedural speaking motion tied to audio playback.

When TTS playback starts:

- `_play_audio_in_thread()` gets the current avatar from the GUI.
- `avatar.start_speaking()` sets `is_speaking = True`.
- audio is played through `AudioProcessor.play_audio()`.
- the thread waits for the generated audio duration plus a short buffer.
- `avatar.stop_speaking()` sets `is_speaking = False`.
- the avatar is reset toward neutral after playback.

While `is_speaking` is true, `update_animation()` overlays mouth movement:

- `mouth_aaa` controls the main mouth opening.
- `mouth_ooo` adds rounded mouth motion.
- `mouth_delta` adds shape variation.

The current formula is sine-based:

```text
talk_cycle = current_time * 12.0
mouth_aaa = 0.2 + sin(talk_cycle) * 0.2
          + sin(talk_cycle * 1.5) * 0.05
          + sin(talk_cycle * 0.5) * 0.05
mouth_ooo = sin(talk_cycle * 0.7) * 0.15 + 0.1
mouth_delta = sin(talk_cycle * 0.9) * 0.1
```

This gives speech motion that changes continuously during playback instead of a static open/closed mouth loop.

There is also a `set_lip_sync()` path that can accept phoneme/duration data and change mouth motion by vowel/consonant type. The stable playback path currently uses the duration-driven speaking overlay for the live desktop loop.

## Audio and Subtitle Coordination

Generated speech uses 24 kHz audio, so duration is computed as:

```text
duration = len(speech) / 24000
```

That duration controls:

- how long the speaking animation stays active
- how long subtitles remain visible
- when processing flags are reset

Audio playback runs in a separate thread. This prevents the desktop GUI from freezing during TTS playback.

## Rendering and Performance Choices

The animation layer makes several practical performance choices:

- uses CUDA when available
- creates pose tensors as `torch.float16`
- wraps poser inference in `torch.no_grad()`
- detaches output and moves it to CPU before wx rendering
- disables AI upscaling by default for performance
- uses fixed 512 x 512 avatar rendering
- scales through wx image operations when needed
- clears CUDA cache periodically during animation
- clears CUDA cache when speaking stops and on cleanup
- uses static rendering continuity when THA4 poser output is unavailable

AI upscaling was intentionally disabled because the extra model cost is not worth it for the real-time avatar loop.

## Paint Strategy

The wx panel uses custom background painting:

- selected background image is scaled to panel size
- if no background exists, a character background color is used
- avatar bitmap is drawn centered with alpha blending
- background erase is suppressed to reduce flicker

The avatar remains a fixed 512 x 512 render target even when the panel size changes. That keeps the model output predictable and avoids changing the THA4 render size frame-to-frame.

## Character Switching

During character switching, the animation layer is recreated around the new `VOICE_TYPE`.

The switch flow:

1. seamless interface updates `VOICE_TYPE`
2. chat GUI selects/recreates the character
3. `AnimatedCharacter` loads the new character model path
4. the new background, color, poser, and animation state are initialized
5. the new character's TTS and Ollama handler are attached separately

This keeps avatar identity, TTS identity, and LLM persona aligned after a switch.
