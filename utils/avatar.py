import wx
import torch
import time
import math
import random
import threading
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


class AnimatedCharacter:
    IMAGE_SIZE = 512

    def __init__(self, parent_window, width: int, height: int, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.width = width
        self.height = height
        
        # Create wx panel
        self.panel = wx.Panel(parent_window, size=(width, height))
        self.panel.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        
        # Bind paint event
        self.panel.Bind(wx.EVT_PAINT, self.on_paint)
        
        # Load character model
        self.character_model = None
        self.poser = None
        self.load_character_model()
        
        # Initialize animations
        self.idle_animation = IdleAnimation()
        self.happy_animation = HappyAnimation()
        self.sad_animation = SadAnimation()
        self.angry_animation = AngryAnimation()
        self.current_animation: Optional[BaseAnimation] = self.idle_animation
        
        # Animation state
        self.is_speaking = False
        self.current_emotion = "neutral"
        self.target_emotion = "neutral"
        self.emotion_blend = 0.0
        self.emotion_transition_speed = 2.0  # Seconds to transition
        
        # Initialize result bitmap
        self.result_bitmap = wx.Bitmap(width, height)
        
        # Store last update time
        self.last_update_time = time.time()
        
        # Start animation timer (30 FPS)
        self.timer = wx.Timer(self.panel)
        self.panel.Bind(wx.EVT_TIMER, self.update_animation)
        self.timer.Start(33)  # ~30 FPS

    def load_character_model(self):
        """Load the character model from YAML config"""
        try:
            model_path = Path("asset/model/ame/character_model/character_model.yaml")
            print(f"Loading character model from: {model_path.absolute()}")
            
            if not model_path.exists():
                print(f"Error: Model file not found at {model_path.absolute()}")
                return
                
            self.character_model = CharacterModel.load(model_path)
            print("Character model loaded")
            
            # Get the character image and poser
            self.character_model.get_character_image(self.device)  # Load image into memory
            self.poser = self.character_model.get_poser(self.device)
            print("Character image and poser loaded successfully")
            
        except Exception as e:
            print(f"Error loading character model: {str(e)}")
            import traceback
            traceback.print_exc()
            self.character_model = None
            self.poser = None

    def update_animation(self, event=None):
        """Update animation state and redraw"""
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time

        # Update emotion blending
        if self.current_emotion != self.target_emotion:
            self.emotion_blend = min(1.0, self.emotion_blend + delta_time / self.emotion_transition_speed)
            if self.emotion_blend >= 1.0:
                self.current_emotion = self.target_emotion
                self.emotion_blend = 0.0
        
        # Get animation parameters
        params = {}
        
        # Get base idle parameters
        idle_params = self.idle_animation.update(delta_time)
        params.update(idle_params)
        
        # Add speaking animation if currently speaking
        if self.is_speaking:
            # Create a natural-looking talking animation
            talk_speed = 0.5  # Extremely slow for observation (was 4.0)
            talk_cycle = (current_time * talk_speed) % 1.0
            
            # Use smoother sine wave for more natural transitions
            mouth_open = abs(math.sin(talk_cycle * math.pi * 2))
            
            # Base mouth parameters - reduced overall to make movements smaller
            base_aaa = 0.25  # Slightly reduced from 0.3 for subtler movement
            base_iii = 0.12  # Slightly reduced from 0.15
            base_ooo = 0.15  # Slightly reduced from 0.2
            
            # Smooth transitions between mouth shapes
            if talk_cycle < 0.3:  # Slightly open
                params["mouth_aaa"] = base_aaa * mouth_open
                params["mouth_iii"] = base_iii * (1 - mouth_open)
            elif talk_cycle < 0.6:  # More open
                params["mouth_aaa"] = base_aaa * mouth_open
                params["mouth_ooo"] = base_ooo * (1 - mouth_open)
            else:  # Transition back to slightly open
                params["mouth_aaa"] = base_aaa * mouth_open * 0.7
                params["mouth_iii"] = base_iii * (1 - mouth_open)
        
        # Blend with emotion animation if transitioning
        if self.emotion_blend > 0:
            if self.target_emotion == "happy":
                emotion_params = self.happy_animation.update(delta_time)
            elif self.target_emotion == "sad":
                emotion_params = self.sad_animation.update(delta_time)
            elif self.target_emotion == "angry":
                emotion_params = self.angry_animation.update(delta_time)
            else:
                emotion_params = {}
            
            # Blend parameters
            for key in emotion_params:
                if key in params:
                    params[key] = params[key] * (1 - self.emotion_blend) + emotion_params[key] * self.emotion_blend
                else:
                    params[key] = emotion_params[key] * self.emotion_blend
        
        # Update character model with parameters
        if self.character_model is not None and self.poser is not None:
            try:
                # Create parameter array
                pose = torch.zeros((1, self.poser.num_parameters), device=self.device)
                
                # Map parameters to their indices
                param_mapping = {
                    # Face rotation
                    "head_x": self.get_parameter_index(PoseParameterCategory.FACE_ROTATION, "head_x"),
                    "head_y": self.get_parameter_index(PoseParameterCategory.FACE_ROTATION, "head_y"),
                    "head_z": self.get_parameter_index(PoseParameterCategory.FACE_ROTATION, "neck_z"),
                    
                    # Body rotation
                    "body_y": self.get_parameter_index(PoseParameterCategory.BODY_ROTATION, "body_y"),
                    "body_z": self.get_parameter_index(PoseParameterCategory.BODY_ROTATION, "body_z"),
                    
                    # Eyes and eyebrows
                    "eye_blink": self.get_parameter_index(PoseParameterCategory.EYE, "eye_blink"),
                    "eye_widen": self.get_parameter_index(PoseParameterCategory.EYE, "eye_widen"),
                    "eye_squint": self.get_parameter_index(PoseParameterCategory.EYE, "eye_squint"),
                    "eye_unimpressed": self.get_parameter_index(PoseParameterCategory.EYE, "eye_unimpressed"),
                    "eye_happy": self.get_parameter_index(PoseParameterCategory.EYE, "eye_happy"),
                    "eye_angry": self.get_parameter_index(PoseParameterCategory.EYE, "eye_angry"),
                    "eye_sad": self.get_parameter_index(PoseParameterCategory.EYE, "eye_sad"),
                    
                    # Iris
                    "iris_small": self.get_parameter_index(PoseParameterCategory.IRIS_MORPH, "iris_small"),
                    "iris_x": self.get_parameter_index(PoseParameterCategory.IRIS_ROTATION, "iris_rotation_x"),
                    "iris_y": self.get_parameter_index(PoseParameterCategory.IRIS_ROTATION, "iris_rotation_y"),
                    
                    # Eyebrows
                    "eyebrow_happy": self.get_parameter_index(PoseParameterCategory.EYEBROW, "eyebrow_happy"),
                    "eyebrow_angry": self.get_parameter_index(PoseParameterCategory.EYEBROW, "eyebrow_angry"),
                    "eyebrow_sad": self.get_parameter_index(PoseParameterCategory.EYEBROW, "eyebrow_sad"),
                    "eyebrow_troubled": self.get_parameter_index(PoseParameterCategory.EYEBROW, "eyebrow_troubled"),
                    
                    # Mouth
                    "mouth_aaa": self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_aaa"),
                    "mouth_iii": self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_iii"),
                    "mouth_uuu": self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_uuu"),
                    "mouth_eee": self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_eee"),
                    "mouth_ooo": self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_ooo"),
                    "mouth_delta": self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_delta"),
                    "mouth_smirk": self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_smirk"),
                    "mouth_lowered_corner": self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_lowered_corner"),
                    
                    # Breathing
                    "breathing": self.get_parameter_index(PoseParameterCategory.BREATHING)
                }
                
                # Set parameters based on mapping
                for param_name, value in params.items():
                    if param_name in param_mapping and param_mapping[param_name] >= 0:
                        index = param_mapping[param_name]
                        if param_name == "eye_blink":  # Special case for eyes to make them blink together
                            pose[0, index] = value
                            pose[0, index + 1] = value
                        else:
                            pose[0, index] = value
                
                # Get character image
                character_image = self.character_model.get_character_image(self.device)
                
                # Generate image using the poser
                with torch.no_grad():
                    output_image = self.poser.pose(character_image, pose)
                    output_image = output_image[0].detach().cpu()
                    output_image = convert_output_image_from_torch_to_numpy(output_image)
                    
                    # Ensure the array is contiguous and in the correct format
                    if len(output_image.shape) == 3:
                        height, width, channels = output_image.shape
                        if channels == 4:  # RGBA format
                            # Ensure the array is contiguous and in the correct order (RGBA)
                            output_image = output_image.copy(order='C')  # Make contiguous in memory
                            
                            # Create wx.Image first
                            wx_image = wx.Image(width, height)
                            wx_image.SetData(output_image[:,:,:3].tobytes())  # Set RGB data
                            wx_image.SetAlpha(output_image[:,:,3].tobytes())  # Set alpha channel separately
                            
                            # Convert to bitmap
                            if width != self.width or height != self.height:
                                wx_image = wx_image.Scale(self.width, self.height, wx.IMAGE_QUALITY_HIGH)
                            
                            self.result_bitmap = wx.Bitmap(wx_image)
                        else:
                            print(f"Unexpected number of channels: {channels}")
                    else:
                        print(f"Unexpected image shape: {output_image.shape}")
            except Exception as e:
                print(f"Error updating animation: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Request redraw
        self.panel.Refresh()

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
            dc.DrawBitmap(self.result_bitmap, 0, 0)

    def set_emotion(self, emotion: str):
        """Set the target emotion to transition to"""
        if emotion != self.target_emotion:
            self.target_emotion = emotion
            self.emotion_blend = 0.0

    def start_speaking(self):
        """Called when character starts speaking"""
        self.is_speaking = True
        
    def stop_speaking(self):
        """Called when character stops speaking"""
        self.is_speaking = False
