import wx
import torch
import time
import math
import random
import threading
import os
from pathlib import Path
from typing import Optional, Dict

from tha4.charmodel.character_model import CharacterModel
from tha4.poser.modes.mode_14 import create_poser
from tha4.poser.poser import PoseParameterCategory
from tha4.image_util import convert_output_image_from_torch_to_numpy
from tha4.app.animations.happy_animation import HappyAnimation
from tha4.app.animations.sad_animation import SadAnimation
from tha4.app.animations.angry_animation import AngryAnimation
from tha4.app.animations.idle_animation import IdleAnimation
from tha4.app.animations.base_animation import BaseAnimation
from tha4.app.animations.surprise_animation import SurpriseAnimation


class AnimatedCharacter:
    IMAGE_SIZE = 512

    def __init__(self, parent_window, width: int, height: int, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), model_path=None):
        self.device = device
        self.width = width
        self.height = height
        self.model_name = os.getenv('VOICE_TYPE', 'Amelia')
        
        # Get background color from bg_color.txt or use model-specific defaults
        bg_color_path = Path(f"asset/model/{self.model_name}/bg_color.txt")
        try:
            with open(bg_color_path, 'r') as f:
                self.bg_color = f.read().strip()
        except:
            # Default background colors for each model
            self.bg_color = "#2b2b3b" if self.model_name == "Eveland" else "#ffd05c"
        
        # Update model path to match the new structure
        self.model_path = model_path or Path(f"asset/model/{self.model_name}/character_model/character_model.yaml")
        
        print(f"Looking for model at: {self.model_path.absolute()}")
        print(f"Using background color: {self.bg_color}")
        
        # Create panel in the parent window with transparent background
        self.panel = wx.Panel(parent_window, size=(width, height))
        self.panel.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)  # Enable custom background drawing
        
        # Bind paint event
        self.panel.Bind(wx.EVT_PAINT, self.on_paint)
        self.panel.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)
        
        # Initialize model-related attributes
        self.character_model = None
        self.poser = None
        self.param_mapping = None  # Initialize param_mapping as None
        self.cached_character_image = None  # Cache for character base image
        self.load_character_model()
        
        # Initialize animations
        self.idle_animation = IdleAnimation()
        self.happy_animation = HappyAnimation()
        self.sad_animation = SadAnimation()
        self.angry_animation = AngryAnimation()
        self.surprise_animation = SurpriseAnimation()
        self.current_animation: Optional[BaseAnimation] = self.idle_animation
        
        # Animation state
        self.is_speaking = False
        self.current_emotion = "neutral"
        self.target_emotion = "neutral"
        self.emotion_blend = 0.0
        self.emotion_blend_speed = 2.0  # Adjust for faster/slower transitions
        
        # Initialize result bitmap
        self.result_bitmap = wx.Bitmap(width, height)
        
        # Store last update time
        self.last_update_time = time.time()
        
        # Memory optimization flags
        self.frame_count = 0
        self.memory_clear_interval = 30  # Clear memory every 30 frames
        
        # Start animation timer (30 FPS)
        self.timer = wx.Timer(self.panel)
        self.panel.Bind(wx.EVT_TIMER, self.update_animation)
        self.timer.Start(67)  

    def load_character_model(self):
        """Load the character model from YAML config"""
        try:
            print(f"Loading character model from: {self.model_path.absolute()}")
            
            if not self.model_path.exists():
                print(f"Error: Model file not found at {self.model_path.absolute()}")
                return
                
            self.character_model = CharacterModel.load(self.model_path)
            print("Character model loaded")
            
            # Cache the character image in memory
            self.cached_character_image = self.character_model.get_character_image(self.device)
            self.poser = self.character_model.get_poser(self.device)
            print("Character image and poser loaded successfully")
            
            # Initialize parameter mapping
            self.param_mapping = {
                # Face rotation
                "head_x": self.get_parameter_index(PoseParameterCategory.FACE_ROTATION, "head_x"),
                "head_y": self.get_parameter_index(PoseParameterCategory.FACE_ROTATION, "head_y"),
                "neck_z": self.get_parameter_index(PoseParameterCategory.FACE_ROTATION, "neck_z"),
                
                # Body rotation
                "body_y": self.get_parameter_index(PoseParameterCategory.BODY_ROTATION, "body_y"),
                "body_z": self.get_parameter_index(PoseParameterCategory.BODY_ROTATION, "body_z"),
                
                # Eyes
                "eye_wink": self.get_parameter_index(PoseParameterCategory.EYE, "eye_wink"),
                "eye_happy_wink": self.get_parameter_index(PoseParameterCategory.EYE, "eye_happy_wink"),
                "eye_surprised": self.get_parameter_index(PoseParameterCategory.EYE, "eye_surprised"),
                "eye_relaxed": self.get_parameter_index(PoseParameterCategory.EYE, "eye_relaxed"),
                "eye_unimpressed": self.get_parameter_index(PoseParameterCategory.EYE, "eye_unimpressed"),
                "eye_raised_lower_eyelid": self.get_parameter_index(PoseParameterCategory.EYE, "eye_raised_lower_eyelid"),
                
                # Iris
                "iris_small": self.get_parameter_index(PoseParameterCategory.IRIS_MORPH, "iris_small"),
                "iris_rotation_x": self.get_parameter_index(PoseParameterCategory.IRIS_ROTATION, "iris_rotation_x"),
                "iris_rotation_y": self.get_parameter_index(PoseParameterCategory.IRIS_ROTATION, "iris_rotation_y"),
                
                # Eyebrows
                "eyebrow_happy": self.get_parameter_index(PoseParameterCategory.EYEBROW, "eyebrow_happy"),
                "eyebrow_angry": self.get_parameter_index(PoseParameterCategory.EYEBROW, "eyebrow_angry"),
                "eyebrow_troubled": self.get_parameter_index(PoseParameterCategory.EYEBROW, "eyebrow_troubled"),
                "eyebrow_lowered": self.get_parameter_index(PoseParameterCategory.EYEBROW, "eyebrow_lowered"),
                "eyebrow_raised": self.get_parameter_index(PoseParameterCategory.EYEBROW, "eyebrow_raised"),
                "eyebrow_serious": self.get_parameter_index(PoseParameterCategory.EYEBROW, "eyebrow_serious"),
                
                # Mouth
                "mouth_aaa": self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_aaa"),
                "mouth_iii": self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_iii"),
                "mouth_uuu": self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_uuu"),
                "mouth_eee": self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_eee"),
                "mouth_ooo": self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_ooo"),
                "mouth_delta": self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_delta"),
                "mouth_lowered_corner": self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_lowered_corner"),
                "mouth_raised_corner": self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_raised_corner"),
                "mouth_smirk": self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_smirk"),
                
                # Breathing
                "breathing": self.get_parameter_index(PoseParameterCategory.BREATHING, "breathing")
            }
            print("Parameter mapping initialized successfully")
            
        except Exception as e:
            print(f"Error loading character model: {str(e)}")
            import traceback
            traceback.print_exc()
            self.character_model = None
            self.poser = None
            self.param_mapping = None
            self.cached_character_image = None

    def update_animation(self, event):
        try:
            current_time = time.time()
            delta_time = current_time - self.last_update_time
            self.last_update_time = current_time

            if self.poser is None:
                return

            # Create pose parameters
            pose = [0.0] * self.poser.num_parameters
            animation_values = None

            # Get animation values based on emotion and maintain it
            if self.target_emotion == "happy":
                animation_values = self.happy_animation.update(delta_time)
                
                # Apply happy-specific parameters
                mouth_smile_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_smile")
                mouth_raised_corner_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_raised_corner")
                mouth_aaa_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_aaa")
                mouth_ooo_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_ooo")
                eyebrow_happy_index = self.get_parameter_index(PoseParameterCategory.EYEBROW, "eyebrow_happy")
                eye_happy_index = self.get_parameter_index(PoseParameterCategory.EYE, "eye_happy")
                
                if mouth_smile_index >= 0:
                    pose[mouth_smile_index] = 1.0
                if mouth_raised_corner_index >= 0:
                    pose[mouth_raised_corner_index] = 0.8
                    pose[mouth_raised_corner_index + 1] = 0.8
                if mouth_aaa_index >= 0:
                    pose[mouth_aaa_index] = 0.6
                if mouth_ooo_index >= 0:
                    pose[mouth_ooo_index] = 0.25
                if eyebrow_happy_index >= 0:
                    pose[eyebrow_happy_index] = 0.6
                    pose[eyebrow_happy_index + 1] = 0.6
                if eye_happy_index >= 0:
                    pose[eye_happy_index] = 0.6
                    pose[eye_happy_index + 1] = 0.6

            elif self.target_emotion == "sad":
                animation_values = self.sad_animation.update(delta_time)
                
                # Handle eye parameters
                eye_unimpressed_index = self.get_parameter_index(PoseParameterCategory.EYE, "eye_unimpressed")
                if eye_unimpressed_index >= 0:
                    pose[eye_unimpressed_index] = 0.8
                    pose[eye_unimpressed_index + 1] = 0.8
                
                # Handle mouth parameters
                mouth_delta_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_delta")
                mouth_lowered_corner_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_lowered_corner")
                if mouth_delta_index >= 0:
                    pose[mouth_delta_index] = -0.3
                if mouth_lowered_corner_index >= 0:
                    pose[mouth_lowered_corner_index] = 0.8
                    pose[mouth_lowered_corner_index + 1] = 0.8
                
                # Handle eyebrow parameters
                eyebrow_troubled_index = self.get_parameter_index(PoseParameterCategory.EYEBROW, "eyebrow_troubled")
                if eyebrow_troubled_index >= 0:
                    pose[eyebrow_troubled_index] = 1.0
                    pose[eyebrow_troubled_index + 1] = 1.0

                # Handle iris parameters
                iris_small_index = self.get_parameter_index(PoseParameterCategory.IRIS_MORPH, "iris_small")
                if iris_small_index >= 0:
                    pose[iris_small_index] = 0.1
                    pose[iris_small_index + 1] = 0.1

                # Add head tilt down for sad emotion
                head_y_index = self.get_parameter_index(PoseParameterCategory.FACE_ROTATION, "head_y")
                if head_y_index >= 0:
                    pose[head_y_index] = -0.3  # Tilt head down

            elif self.target_emotion == "angry":
                animation_values = self.angry_animation.update(delta_time)
                
                # Handle eye parameters
                eye_unimpressed_index = self.get_parameter_index(PoseParameterCategory.EYE, "eye_unimpressed")
                if eye_unimpressed_index >= 0:
                    pose[eye_unimpressed_index] = 0.35
                    pose[eye_unimpressed_index + 1] = 0.35
                
                # Handle mouth parameters
                mouth_delta_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_delta")
                mouth_lowered_corner_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_lowered_corner")
                if mouth_delta_index >= 0:
                    pose[mouth_delta_index] = 0.7
                if mouth_lowered_corner_index >= 0:
                    pose[mouth_lowered_corner_index] = 0.6
                    pose[mouth_lowered_corner_index + 1] = 0.6
                
                # Handle eyebrow parameters
                eyebrow_angry_index = self.get_parameter_index(PoseParameterCategory.EYEBROW, "eyebrow_angry")
                if eyebrow_angry_index >= 0:
                    pose[eyebrow_angry_index] = 0.8
                    pose[eyebrow_angry_index + 1] = 0.8

            elif self.target_emotion == "surprise":
                animation_values = self.surprise_animation.update(delta_time)
                
                # Handle eye parameters
                eye_surprised_index = self.get_parameter_index(PoseParameterCategory.EYE, "eye_surprised")
                if eye_surprised_index >= 0:
                    pose[eye_surprised_index] = 0.9
                    pose[eye_surprised_index + 1] = 0.9
                
                # Handle eyebrow parameters
                eyebrow_raised_index = self.get_parameter_index(PoseParameterCategory.EYEBROW, "eyebrow_raised")
                if eyebrow_raised_index >= 0:
                    pose[eyebrow_raised_index] = 0.8
                    pose[eyebrow_raised_index + 1] = 0.8
                
                # Handle iris parameters
                iris_small_index = self.get_parameter_index(PoseParameterCategory.IRIS_MORPH, "iris_small")
                if iris_small_index >= 0:
                    pose[iris_small_index] = -0.2
                    pose[iris_small_index + 1] = -0.2

            else:  # neutral/idle
                animation_values = self.idle_animation.update(delta_time)

            # Apply common animation values
            if animation_values:
                # Apply breathing
                breathing_index = self.get_parameter_index(PoseParameterCategory.BREATHING)
                if breathing_index >= 0:
                    pose[breathing_index] = animation_values.get('breathing', 0.0)
                
                # Set head rotation
                head_x_index = self.get_parameter_index(PoseParameterCategory.FACE_ROTATION, "head_x")
                head_y_index = self.get_parameter_index(PoseParameterCategory.FACE_ROTATION, "head_y")
                neck_z_index = self.get_parameter_index(PoseParameterCategory.FACE_ROTATION, "neck_z")
                if head_x_index >= 0:
                    pose[head_x_index] = animation_values.get('head_x', 0.0)
                if head_y_index >= 0:
                    pose[head_y_index] = animation_values.get('head_y', 0.0)
                if neck_z_index >= 0:
                    pose[neck_z_index] = animation_values.get('neck_z', 0.0)

                # Set body rotation
                body_y_index = self.get_parameter_index(PoseParameterCategory.BODY_ROTATION, "body_y")
                body_z_index = self.get_parameter_index(PoseParameterCategory.BODY_ROTATION, "body_z")
                if body_y_index >= 0:
                    pose[body_y_index] = animation_values.get('body_y', 0.0)
                if body_z_index >= 0:
                    pose[body_z_index] = animation_values.get('body_z', 0.0)

                # Set eye rotation
                iris_rotation_x_index = self.get_parameter_index(PoseParameterCategory.IRIS_ROTATION, "iris_rotation_x")
                iris_rotation_y_index = self.get_parameter_index(PoseParameterCategory.IRIS_ROTATION, "iris_rotation_y")
                if iris_rotation_x_index >= 0:
                    pose[iris_rotation_x_index] = animation_values.get('iris_rotation_x', 0.0)
                    pose[iris_rotation_x_index + 1] = animation_values.get('iris_rotation_x', 0.0)
                if iris_rotation_y_index >= 0:
                    pose[iris_rotation_y_index] = animation_values.get('iris_rotation_y', 0.0)
                    pose[iris_rotation_y_index + 1] = animation_values.get('iris_rotation_y', 0.0)

                # Set blinking
                eye_wink_index = self.get_parameter_index(PoseParameterCategory.EYE, "eye_wink")
                if eye_wink_index >= 0 and 'eye_wink' in animation_values:
                    wink_value = animation_values['eye_wink']
                    pose[eye_wink_index] = wink_value
                    pose[eye_wink_index + 1] = wink_value

            # Enhanced mouth movement when speaking
            if self.is_speaking:
                mouth_aaa_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_aaa")
                mouth_ooo_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_ooo")
                mouth_delta_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_delta")
                
                # Create more natural mouth movement
                talk_speed = 12.0
                talk_cycle = current_time * talk_speed
                
                # Main mouth opening
                if mouth_aaa_index >= 0:
                    mouth_open = (
                        0.2 +
                        math.sin(talk_cycle) * 0.2 +
                        math.sin(talk_cycle * 1.5) * 0.05 +
                        math.sin(talk_cycle * 0.5) * 0.05
                    )
                    mouth_open = max(0.1, min(0.5, mouth_open))
                    pose[mouth_aaa_index] = mouth_open
                
                if mouth_ooo_index >= 0:
                    mouth_round = math.sin(talk_cycle * 0.7) * 0.15 + 0.1
                    pose[mouth_ooo_index] = mouth_round
                
                if mouth_delta_index >= 0:
                    mouth_shape = math.sin(talk_cycle * 0.9) * 0.1
                    pose[mouth_delta_index] = mouth_shape

            # Convert pose list to tensor
            pose_tensor = torch.tensor([pose], device=self.device, dtype=torch.float16)
            
            # Generate and display image
            self._generate_output_image(pose_tensor)

        except Exception as e:
            print(f"Error in animation update: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_parameter_index(self, category: PoseParameterCategory, param_name: str = None) -> int:
        """Get the parameter index for a given category and name"""
        params = self.poser.get_pose_parameter_groups()
        for param in params:
            if param.get_category() == category:
                if param_name is None or param.get_group_name() == param_name:
                    return param.get_parameter_index()
        return -1

    def on_paint(self, event):
        """Handle paint event"""
        dc = wx.BufferedPaintDC(self.panel)
        if self.result_bitmap.IsOk():
            # First draw the background
            dc.SetBackground(wx.Brush(self.bg_color))
            dc.Clear()
            # Then draw the character bitmap
            dc.DrawBitmap(self.result_bitmap, 0, 0)

    def on_erase_background(self, event):
        """Handle background erasure - do nothing to prevent flicker"""
        pass

    def set_emotion(self, emotion: str):
        """Set the target emotion for animation transition"""
        print(f"[DEBUG] Raw emotion from classifier: {emotion}")
        
        # Map RoBERTa emotions to our animation states
        emotion_mapping = {
            # Happy emotions
            "admiration": "happy",
            "amusement": "happy",
            "approval": "happy",
            "excitement": "happy",
            "gratitude": "happy",
            "joy": "happy",
            "love": "happy",
            "optimism": "happy",
            "pride": "happy",
            
            # Sad emotions
            "disappointment": "sad",
            "grief": "sad",
            "remorse": "sad",
            "sadness": "sad",
            
            # Angry emotions
            "anger": "angry",
            "annoyance": "angry",
            "disapproval": "angry",
            "disgust": "angry",
            
            # Surprised emotions
            "realization": "surprise",
            "surprise": "surprise",
            
            # Default to current emotion unless explicitly requested
            "neutral": self.target_emotion,  # Keep current emotion instead of going neutral
            "confusion": self.target_emotion,
            "caring": self.target_emotion,
            "curiosity": self.target_emotion,
            "desire": self.target_emotion,
            "relief": self.target_emotion
        }
        
        # Skip emotion update if it's an automatic neutral after speaking
        if emotion.lower() == "neutral" and self.target_emotion != "neutral":
            print(f"[DEBUG] Keeping current emotion: {self.target_emotion}")
            return
        
        # Only change emotion if explicitly mapped (keeps current emotion as default)
        mapped_emotion = emotion_mapping.get(emotion.lower(), self.target_emotion)
        
        # Only update if emotion actually changes
        if mapped_emotion != self.target_emotion:
            self.target_emotion = mapped_emotion
            print(f"[DEBUG] Setting emotion to: {self.target_emotion}")

    def start_speaking(self):
        """Called when character starts speaking"""
        self.is_speaking = True
        
    def stop_speaking(self):
        """Called when character stops speaking"""
        self.is_speaking = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate_output_image(self, pose):
        with torch.no_grad():
            output_image = self.poser.pose(self.cached_character_image, pose)
            output_image = output_image[0].detach().cpu()
            output_image = convert_output_image_from_torch_to_numpy(output_image)
            
            # Convert to bitmap
            if len(output_image.shape) == 3 and output_image.shape[2] == 4:
                output_image = output_image.copy(order='C')
                wx_image = wx.Image(output_image.shape[1], output_image.shape[0])
                wx_image.SetData(output_image[:,:,:3].tobytes())
                wx_image.SetAlpha(output_image[:,:,3].tobytes())  # Preserve alpha channel
                
                if output_image.shape[1] != self.width or output_image.shape[0] != self.height:
                    wx_image = wx_image.Scale(self.width, self.height, wx.IMAGE_QUALITY_HIGH)
                
                self.result_bitmap = wx.Bitmap(wx_image)
                
                # Periodic memory cleanup
                if self.frame_count == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Clear intermediate tensors
                del output_image
                del pose
                
                # Request redraw
                self.panel.Refresh()
            else:
                print(f"Error: Invalid image format - shape: {output_image.shape}")

    def set_lip_sync(self, phonemes, durations):
        """Set lip sync data for TTS audio"""
        self.lip_sync_data = []
        current_time = 0
        
        # Convert phonemes and durations into timing data
        for phoneme, duration in zip(phonemes.split(), durations):
            # Determine if this phoneme should cause mouth movement
            is_vowel = phoneme.lower() in 'aeiou'
            is_consonant = not is_vowel and phoneme.isalpha()
            
            # Add mouth movement data
            self.lip_sync_data.append({
                'start_time': current_time,
                'duration': float(duration),
                'is_vowel': is_vowel,
                'is_consonant': is_consonant
            })
            current_time += float(duration)
        
        self.lip_sync_start_time = time.time()
        self.is_speaking = True

    def _apply_lip_sync(self, pose, current_time):
        """Apply simplified dynamic lip sync"""
        if not self.is_speaking or not hasattr(self, 'lip_sync_data'):
            return
        
        elapsed_time = current_time - self.lip_sync_start_time
        
        # Find current phoneme timing
        current_phoneme = None
        for phoneme_data in self.lip_sync_data:
            if phoneme_data['start_time'] <= elapsed_time < (phoneme_data['start_time'] + phoneme_data['duration']):
                current_phoneme = phoneme_data
                break
        
        if current_phoneme:
            # Get mouth parameter indices
            mouth_aaa_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_aaa")
            mouth_ooo_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_ooo")
            
            # Calculate mouth opening based on phoneme type
            if current_phoneme['is_vowel']:
                # Wider mouth opening for vowels
                mouth_open_amount = 0.7 + math.sin(current_time * 8.0) * 0.1  # Add slight variation
                if mouth_aaa_index >= 0:
                    pose[0, mouth_aaa_index] = mouth_open_amount
            elif current_phoneme['is_consonant']:
                # Smaller mouth opening for consonants
                mouth_open_amount = 0.3 + math.sin(current_time * 8.0) * 0.05  # Add slight variation
                if mouth_ooo_index >= 0:
                    pose[0, mouth_ooo_index] = mouth_open_amount
            
            # Add natural mouth movement variation
            variation = math.sin(current_time * 12.0) * 0.05  # Faster, subtle variation
            if mouth_aaa_index >= 0:
                pose[0, mouth_aaa_index] += variation
        
        # Check if speech is finished
        if elapsed_time > self.lip_sync_data[-1]['start_time'] + self.lip_sync_data[-1]['duration']:
            self.is_speaking = False

    def cleanup(self):
        """Clean up resources and stop animation timer"""
        if self.timer and self.timer.IsRunning():
            self.timer.Stop()
        if hasattr(self, 'panel'):
            self.panel.Destroy()
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def start_animation(self):
        """Start the animation timer"""
        if not self.timer:
            self.timer = wx.Timer(self.panel)
            self.panel.Bind(wx.EVT_TIMER, self.update_animation)
        if not self.timer.IsRunning():
            self.timer.Start(67)  # ~15 FPS

    def stop_animation(self):
        """Stop the animation timer"""
        if self.timer and self.timer.IsRunning():
            self.timer.Stop()

class IdleAnimation:
    def __init__(self):
        # Breathing
        self.breathing_cycle = 0.0  # 0.0 to 1.0
        self.breathing_speed = 0.5  # Full cycle per second
        
        # Blinking
        self.blink_cycle = 0.0  # 0.0 to 1.0
        self.blink_speed = 0.25  # Time for one blink
        self.time_until_next_blink = random.uniform(2.0, 4.0)
        
        # Enhanced head movement
        self.head_x_cycle = 0.0  # Side to side
        self.head_y_cycle = 0.0  # Up and down
        self.neck_z_cycle = 0.0  # Tilt
        self.head_movement_speed = 0.15
        self.head_movement_amount = 0.4
        
        # Enhanced body movement
        self.body_y_cycle = 0.0  # Side to side
        self.body_z_cycle = 0.0  # Forward/backward
        self.body_movement_speed = 0.15
        self.body_movement_amount = 0.35
        
        # Add slight body tilt for more natural sway
        self.body_tilt_cycle = 0.0
        self.body_tilt_speed = 0.15
        self.body_tilt_amount = 0.2
        
        # Add slight offset to make movement more interesting
        self.body_y_offset = random.uniform(-0.1, 0.1)
        self.body_z_offset = random.uniform(-0.1, 0.1)
        
        # Eye movement
        self.iris_rotation_x_cycle = 0.0
        self.iris_rotation_y_cycle = 0.0
        self.iris_movement_speed = 0.08
        self.iris_movement_amount = 0.4
        self.time_until_next_eye_movement = random.uniform(2.0, 4.0)
        self.current_eye_target = (0, 0)
        self.next_eye_target = (0, 0)
        self.eye_movement_progress = 1.0

    def update(self, delta_time):
        # Update cycles
        self.breathing_cycle = (self.breathing_cycle + delta_time * self.breathing_speed) % 1.0
        self.head_x_cycle = (self.head_x_cycle + delta_time * self.head_movement_speed) % 1.0
        self.head_y_cycle = (self.head_y_cycle + delta_time * self.head_movement_speed) % 1.0
        self.neck_z_cycle = (self.neck_z_cycle + delta_time * self.head_movement_speed) % 1.0
        
        # Update blink
        self.time_until_next_blink -= delta_time
        if self.time_until_next_blink <= 0:
            self.blink_cycle = 0.0
            self.time_until_next_blink = random.uniform(2.0, 4.0)
        elif self.blink_cycle < 1.0:
            self.blink_cycle = min(1.0, self.blink_cycle + delta_time / self.blink_speed)
        
        # Calculate values
        breathing = math.sin(self.breathing_cycle * math.pi * 2) * 0.3
        head_x = math.sin(self.head_x_cycle * math.pi * 2) * self.head_movement_amount
        head_y = math.sin(self.head_y_cycle * math.pi * 2) * self.head_movement_amount
        neck_z = math.sin(self.neck_z_cycle * math.pi * 2) * self.head_movement_amount
        
        # Calculate blink value
        blink = 0.0
        if self.blink_cycle < 0.5:
            blink = math.sin(self.blink_cycle * math.pi * 2) * 0.8
        
        return {
            'breathing': breathing,
            'head_x': head_x,
            'head_y': head_y,
            'neck_z': neck_z,
            'eye_wink': max(0.0, blink),
            'iris_rotation_x': 0.0,
            'iris_rotation_y': 0.0,
            'body_y': 0.0,
            'body_z': 0.0
        }
