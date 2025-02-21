import os
import sys

# Add the project root directory to Python path when run directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(project_root)

import logging
import time
import math
import random
from typing import List, Optional

import torch
import wx
import numpy as np

from tha4.charmodel.character_model import CharacterModel
from tha4.poser.modes.mode_14 import create_poser
from tha4.poser.poser import PoseParameterCategory
from tha4.image_util import convert_output_image_from_torch_to_numpy
from tha4.app.animations.happy_animation import HappyAnimation
from tha4.app.animations.sad_animation import SadAnimation
from tha4.app.animations.angry_animation import AngryAnimation
from tha4.app.animations.surprise_animation import SurpriseAnimation

class AnimationPreset:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.is_active = False

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
        self.neck_z_cycle = 0.0  # Tilt (renamed from head_z to neck_z)
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
        self.iris_rotation_x_cycle = 0.0  # Left-right eye movement (renamed from iris_x)
        self.iris_rotation_y_cycle = 0.0  # Up-down eye movement (renamed from iris_y)
        self.iris_movement_speed = 0.08
        self.iris_movement_amount = 0.4
        self.time_until_next_eye_movement = random.uniform(2.0, 4.0)
        self.current_eye_target = (0, 0)
        self.next_eye_target = (0, 0)
        self.eye_movement_progress = 1.0
        
        # Look target system
        self.current_look_target = (0, 0)
        self.next_look_target = (0, 0)
        self.look_transition_progress = 1.0
        self.time_until_next_look = random.uniform(1.5, 3.0)
        self.look_change_duration = 1.0  # Slightly faster transitions
        
        # Head following parameters
        self.head_follow_amount = 0.8  # Increased head movement
        self.head_lag = 0.25  # Slightly quicker response
        
        # Interest point system - more varied looking points
        self.interest_points = [
            (-0.8, 0.3),     # Far upper left
            (0.8, 0.3),      # Far upper right
            (-0.8, 0),       # Far left
            (0.8, 0),        # Far right
            (0, 0.4),        # Straight up
            (-0.5, 0.35),    # Upper left
            (0.5, 0.35),     # Upper right
            (-0.5, -0.1),    # Lower left
            (0.5, -0.1),     # Lower right
            (0, -0.2),       # Down
            # Add some middle points
            (-0.4, 0.2),     # Mid upper left
            (0.4, 0.2),      # Mid upper right
            (-0.4, 0),       # Mid left
            (0.4, 0),        # Mid right
        ]
        
        # Point weights - make some positions more likely
        self.point_weights = [
            2.0,  # Far upper left
            2.0,  # Far upper right
            1.5,  # Far left
            1.5,  # Far right
            1.0,  # Straight up
            2.0,  # Upper left
            2.0,  # Upper right
            1.0,  # Lower left
            1.0,  # Lower right
            0.5,  # Down
            1.5,  # Mid upper left
            1.5,  # Mid upper right
            1.0,  # Mid left
            1.0,  # Mid right
        ]
        
        # Gentle sway
        self.sway_cycle = 0.0
        self.sway_speed = 0.2  # Very slow sway
        self.sway_amount = 0.05  # Subtle amount

    def update(self, delta_time: float) -> dict:
        # Update breathing cycle
        self.breathing_cycle = (self.breathing_cycle + delta_time * self.breathing_speed) % (math.pi * 2)
        breathing_value = math.sin(self.breathing_cycle) * 0.5 + 0.5

        # Update head movement cycles
        self.head_x_cycle = (self.head_x_cycle + delta_time * self.head_movement_speed) % (math.pi * 2)
        self.head_y_cycle = (self.head_y_cycle + delta_time * self.head_movement_speed * 0.7) % (math.pi * 2)
        self.neck_z_cycle = (self.neck_z_cycle + delta_time * self.head_movement_speed * 0.5) % (math.pi * 2)

        # Calculate head movement
        head_x = math.sin(self.head_x_cycle) * self.head_movement_amount
        head_y = math.sin(self.head_y_cycle) * self.head_movement_amount
        neck_z = math.sin(self.neck_z_cycle) * self.head_movement_amount * 0.5

        # Update body movement cycles
        self.body_y_cycle = (self.body_y_cycle + delta_time * self.body_movement_speed) % (math.pi * 2)
        self.body_z_cycle = (self.body_z_cycle + delta_time * self.body_movement_speed * 0.7) % (math.pi * 2)

        # Calculate body movement
        body_y = math.sin(self.body_y_cycle) * self.body_movement_amount + self.body_y_offset
        body_z = math.sin(self.body_z_cycle) * self.body_movement_amount * 0.6 + self.body_z_offset

        # Update eye movement
        self.iris_rotation_x_cycle = (self.iris_rotation_x_cycle + delta_time * self.iris_movement_speed) % (math.pi * 2)
        self.iris_rotation_y_cycle = (self.iris_rotation_y_cycle + delta_time * self.iris_movement_speed * 0.7) % (math.pi * 2)
        
        # Calculate eye movement
        iris_rotation_x = math.sin(self.iris_rotation_x_cycle) * self.iris_movement_amount
        iris_rotation_y = math.sin(self.iris_rotation_y_cycle) * self.iris_movement_amount

        # Update blinking
        self.time_until_next_blink -= delta_time
        if self.time_until_next_blink <= 0:
            self.blink_cycle = 0.0
            self.time_until_next_blink = random.uniform(2.0, 4.0)
        
        if self.blink_cycle < 1.0:
            self.blink_cycle = min(1.0, self.blink_cycle + delta_time / self.blink_speed)
        
        eye_wink = math.sin(self.blink_cycle * math.pi) if self.blink_cycle < 1.0 else 0.0

        return {
            'breathing': breathing_value,
            'head_x': head_x,
            'head_y': head_y,
            'neck_z': neck_z,
            'body_y': body_y,
            'body_z': body_z,
            'iris_rotation_x': iris_rotation_x,
            'iris_rotation_y': iris_rotation_y,
            'eye_wink': eye_wink
        }

class AutonomousAnimationFrame(wx.Frame):
    IMAGE_SIZE = 512
    NUM_PARAMETERS = 45

    def __init__(self, device: torch.device):
        super().__init__(None, wx.ID_ANY, "Autonomous Animation")
        self.device = device
        self.character_model = None
        self.poser = None
        self.torch_source_image = None
        
        # Animation parameters
        self.idle_animation = IdleAnimation()
        self.happy_animation = HappyAnimation()
        self.sad_animation = SadAnimation()
        self.angry_animation = AngryAnimation()
        self.surprise_animation = SurpriseAnimation()
        
        # Animation presets
        self.animation_presets = {
            'idle_breathing': AnimationPreset(
                "Idle Animation", 
                "Basic idle animation with natural movements"
            ),
            'happy': AnimationPreset(
                "Happy", 
                "Cheerful, energetic animation with bouncy movements"
            ),
            'sad': AnimationPreset(
                "Sad", 
                "Drooping, slow movements with occasional long blinks"
            ),
            'angry': AnimationPreset(
                "Angry", 
                "Tense, quick movements with intense expressions"
            ),
            'surprise': AnimationPreset(
                "Surprise",
                "Sudden wide-eyed expression with raised eyebrows"
            ),
        }
        
        self.last_update_time = time.time()
        
        # Initialize UI
        self.init_ui()
        
        # Initialize result_bitmap
        self.result_bitmap = wx.Bitmap(self.IMAGE_SIZE, self.IMAGE_SIZE)
        
        # Timer with slightly lower frequency
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update_animation, self.timer)
        self.timer.Start(33)  # ~30 FPS instead of 60 FPS
        
        # Bind window close event
        self.Bind(wx.EVT_CLOSE, self.on_close)

    def init_ui(self):
        self.main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(self.main_sizer)
        
        # Left panel for image
        self.init_left_panel()
        
        # Right panel for controls
        self.init_right_panel()
        
        # Set window size
        self.SetSize(self.IMAGE_SIZE + 250, self.IMAGE_SIZE + 100)

    def init_left_panel(self):
        self.left_panel = wx.Panel(self)
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        self.left_panel.SetSizer(left_sizer)
        
        # Image display panel with double buffering
        self.image_panel = wx.Panel(
            self.left_panel, 
            size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
            style=wx.FULL_REPAINT_ON_RESIZE | wx.BORDER_NONE)
        self.image_panel.SetBackgroundStyle(wx.BG_STYLE_PAINT)  # Enable double buffering
        self.image_panel.Bind(wx.EVT_PAINT, self.paint_result)
        left_sizer.Add(self.image_panel, 0, wx.ALL, 5)
        
        # Load model button
        self.load_button = wx.Button(self.left_panel, label="Load Model")
        self.load_button.Bind(wx.EVT_BUTTON, self.load_model)
        left_sizer.Add(self.load_button, 0, wx.ALL | wx.EXPAND, 5)
        
        # Add loading text
        self.loading_text = wx.StaticText(self.left_panel, label="Loading...", style=wx.ALIGN_CENTER)
        self.loading_text.SetFont(wx.Font(18, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        left_sizer.Add(self.loading_text, 0, wx.ALL | wx.ALIGN_CENTER, 5)
        self.loading_text.Hide()  # Hide initially
        
        # Add left panel to main sizer here
        self.main_sizer.Add(self.left_panel, 0, wx.ALL, 5)

    def init_right_panel(self):
        self.right_panel = wx.Panel(self)
        right_sizer = wx.BoxSizer(wx.VERTICAL)
        self.right_panel.SetSizer(right_sizer)
        
        # Title
        title = wx.StaticText(self.right_panel, label="Animation Presets")
        title.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        right_sizer.Add(title, 0, wx.ALL | wx.ALIGN_CENTER, 5)
        
        # Separator
        right_sizer.Add(wx.StaticLine(self.right_panel), 0, wx.EXPAND | wx.ALL, 5)
        
        # Animation preset buttons
        self.preset_buttons = {}
        for preset_id, preset in self.animation_presets.items():
            btn = wx.ToggleButton(self.right_panel, label=preset.name)
            btn.SetValue(preset.is_active)
            btn.Bind(wx.EVT_TOGGLEBUTTON, 
                    lambda evt, pid=preset_id: self.on_preset_toggle(evt, pid))
            
            # Add description text
            if preset.description:
                desc = wx.StaticText(self.right_panel, label=preset.description)
                desc.Wrap(200)  # Wrap text at 200 pixels
                
                # Create a vertical sizer for this preset
                preset_sizer = wx.BoxSizer(wx.VERTICAL)
                preset_sizer.Add(btn, 0, wx.EXPAND | wx.ALL, 2)
                preset_sizer.Add(desc, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)
                right_sizer.Add(preset_sizer, 0, wx.EXPAND | wx.ALL, 5)
            else:
                right_sizer.Add(btn, 0, wx.EXPAND | wx.ALL, 5)
            
            self.preset_buttons[preset_id] = btn
        
        self.main_sizer.Add(self.right_panel, 0, wx.ALL | wx.EXPAND, 5)

    def on_preset_toggle(self, event, preset_id):
        button = event.GetEventObject()
        is_active = button.GetValue()
        
        # Update preset state
        self.animation_presets[preset_id].is_active = is_active
        
        # For now, we only have one preset, so no need for complex logic
        # When we add more presets, we'll need to handle combinations here

    def load_model(self, event):
        dir_name = "data/character_models"
        with wx.FileDialog(
            self, "Choose a model", dir_name, "", "*.yaml",
            wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as file_dialog:
            
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return
                
            try:
                # Show loading state
                self.loading_text.Show()
                self.image_panel.Hide()
                self.Refresh()
                
                model_file = os.path.join(file_dialog.GetDirectory(), 
                                        file_dialog.GetFilename())
                self.character_model = CharacterModel.load(model_file)
                self.torch_source_image = self.character_model.get_character_image(
                    self.device)
                self.poser = self.character_model.get_poser(self.device)
                
                # Initial render
                self.loading_text.Hide()
                self.image_panel.Show()
                self.update_image([0.0] * self.NUM_PARAMETERS)  # Render neutral pose
            except Exception as e:
                wx.MessageBox(f"Could not load model: {str(e)}", "Error", 
                            wx.OK | wx.ICON_ERROR)
                self.loading_text.Hide()
                self.image_panel.Show()

    def update_animation(self, event):
        if self.poser is None:
            return
            
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time

        # Create pose parameters
        pose = [0.0] * self.NUM_PARAMETERS
        
        # Get animation values based on active preset
        if self.animation_presets['idle_breathing'].is_active:
            animation_values = self.idle_animation.update(delta_time)
        elif self.animation_presets['happy'].is_active:
            animation_values = self.happy_animation.update(delta_time)
            # Apply happy-specific parameters
            mouth_smile_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_smile")
            if mouth_smile_index >= 0:
                pose[mouth_smile_index] = animation_values.get('mouth_happy', 1.0)
        elif self.animation_presets['sad'].is_active:
            animation_values = self.sad_animation.update(delta_time)
            
            # Handle mouth parameters
            mouth_aaa_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_aaa")
            mouth_ooo_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_ooo")
            mouth_delta_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_delta")
            mouth_lowered_corner_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_lowered_corner")
            
            if mouth_aaa_index >= 0:
                pose[mouth_aaa_index] = animation_values.get('mouth_aaa', 0.2)
            if mouth_ooo_index >= 0:
                pose[mouth_ooo_index] = animation_values.get('mouth_ooo', 0.4)
            if mouth_delta_index >= 0:
                pose[mouth_delta_index] = animation_values.get('mouth_delta', -0.3)
            if mouth_lowered_corner_index >= 0:
                pose[mouth_lowered_corner_index] = animation_values.get('mouth_lowered_corner_left', 0.8)
                pose[mouth_lowered_corner_index + 1] = animation_values.get('mouth_lowered_corner_right', 0.8)
            
            # Handle troubled eyebrows
            eyebrow_troubled_index = self.get_parameter_index(PoseParameterCategory.EYEBROW, "eyebrow_troubled")
            if eyebrow_troubled_index >= 0:
                pose[eyebrow_troubled_index] = animation_values.get('eyebrow_troubled_left', 1.0)
                pose[eyebrow_troubled_index + 1] = animation_values.get('eyebrow_troubled_right', 1.0)
            
            # Handle eye parameters
            eye_unimpressed_index = self.get_parameter_index(PoseParameterCategory.EYE, "eye_unimpressed")
            if eye_unimpressed_index >= 0:
                pose[eye_unimpressed_index] = animation_values.get('eye_unimpressed', 0.8)
                pose[eye_unimpressed_index + 1] = animation_values.get('eye_unimpressed', 0.8)  # Set right eye
            
            # Handle head rotation
            head_x_index = self.get_parameter_index(PoseParameterCategory.FACE_ROTATION, "head_x")
            if head_x_index >= 0:
                pose[head_x_index] = animation_values.get('head_x', -0.5)  # Ensure negative value for down tilt
            
            # Handle iris parameters for sad animation - keep centered
            iris_small_index = self.get_parameter_index(PoseParameterCategory.IRIS_MORPH, "iris_small")
            iris_rotation_x_index = self.get_parameter_index(PoseParameterCategory.IRIS_ROTATION, "iris_rotation_x")
            iris_rotation_y_index = self.get_parameter_index(PoseParameterCategory.IRIS_ROTATION, "iris_rotation_y")
            
            if iris_small_index >= 0:
                pose[iris_small_index] = animation_values.get('iris_small', 0.1)
                pose[iris_small_index + 1] = animation_values.get('iris_small', 0.1)
            
            if iris_rotation_x_index >= 0:
                # Set both eyes' x rotation to 0 (centered)
                pose[iris_rotation_x_index] = 0.0
                pose[iris_rotation_x_index + 1] = 0.0
            
            if iris_rotation_y_index >= 0:
                pose[iris_rotation_y_index] = animation_values.get('iris_rotation_y', -0.6)
                pose[iris_rotation_y_index + 1] = animation_values.get('iris_rotation_y', -0.6)
        elif self.animation_presets['angry'].is_active:
            animation_values = self.angry_animation.update(delta_time)
            
            # Handle mouth parameters
            mouth_delta_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_delta")
            mouth_lowered_corner_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_lowered_corner")
            
            if mouth_delta_index >= 0:
                pose[mouth_delta_index] = animation_values.get('mouth_delta', 0.7)
            if mouth_lowered_corner_index >= 0:
                pose[mouth_lowered_corner_index] = animation_values.get('mouth_lowered_corner_left', 0.6)
                pose[mouth_lowered_corner_index + 1] = animation_values.get('mouth_lowered_corner_right', 0.6)
            
            # Handle eyebrow parameters
            eyebrow_angry_index = self.get_parameter_index(PoseParameterCategory.EYEBROW, "eyebrow_angry")
            if eyebrow_angry_index >= 0:
                pose[eyebrow_angry_index] = animation_values.get('eyebrow_angry_left', 0.8)
                pose[eyebrow_angry_index + 1] = animation_values.get('eyebrow_angry_right', 0.8)
            
            # Handle eye parameters
            eye_unimpressed_index = self.get_parameter_index(PoseParameterCategory.EYE, "eye_unimpressed")
            if eye_unimpressed_index >= 0:
                pose[eye_unimpressed_index] = animation_values.get('eye_unimpressed_left', 0.35)
                pose[eye_unimpressed_index + 1] = animation_values.get('eye_unimpressed_right', 0.35)
            
            # Handle mouth parameters
            mouth_smirk_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_smirk")
            if mouth_smirk_index >= 0:
                pose[mouth_smirk_index] = animation_values.get('mouth_smirk', 1.0)
            
            # Handle iris parameters for angry animation - keep static
            iris_small_index = self.get_parameter_index(PoseParameterCategory.IRIS_MORPH, "iris_small")
            iris_rotation_x_index = self.get_parameter_index(PoseParameterCategory.IRIS_ROTATION, "iris_rotation_x")
            iris_rotation_y_index = self.get_parameter_index(PoseParameterCategory.IRIS_ROTATION, "iris_rotation_y")
            
            if iris_small_index >= 0:
                pose[iris_small_index] = 0.0  # Default size for both eyes
                pose[iris_small_index + 1] = 0.0
            
            if iris_rotation_x_index >= 0:
                pose[iris_rotation_x_index] = 0.0  # Keep centered horizontally
            
            if iris_rotation_y_index >= 0:
                pose[iris_rotation_y_index] = 0.0  # Keep centered vertically
            
            # Handle head rotation
            head_x_index = self.get_parameter_index(PoseParameterCategory.FACE_ROTATION, "head_x")
            if head_x_index >= 0:
                pose[head_x_index] = animation_values.get('head_x', -0.5)  # Ensure negative value for down tilt
        elif self.animation_presets['surprise'].is_active:
            animation_values = self.surprise_animation.update(delta_time)
            
            # Handle eye parameters
            eye_surprised_index = self.get_parameter_index(PoseParameterCategory.EYE, "eye_surprised")
            if eye_surprised_index >= 0:
                pose[eye_surprised_index] = animation_values.get('eye_surprised', 0.9)
                pose[eye_surprised_index + 1] = animation_values.get('eye_surprised', 0.9)
            
            # Handle eyebrow parameters
            eyebrow_raised_index = self.get_parameter_index(PoseParameterCategory.EYEBROW, "eyebrow_raised")
            if eyebrow_raised_index >= 0:
                pose[eyebrow_raised_index] = animation_values.get('eyebrow_raised', 0.8)
                pose[eyebrow_raised_index + 1] = animation_values.get('eyebrow_raised', 0.8)
            
            # Handle mouth parameters
            mouth_aaa_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_aaa")
            mouth_ooo_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_ooo")
            if mouth_aaa_index >= 0:
                pose[mouth_aaa_index] = animation_values.get('mouth_aaa', 0.7)
            if mouth_ooo_index >= 0:
                pose[mouth_ooo_index] = animation_values.get('mouth_ooo', 0.3)
            
            # Handle iris parameters
            iris_small_index = self.get_parameter_index(PoseParameterCategory.IRIS_MORPH, "iris_small")
            if iris_small_index >= 0:
                pose[iris_small_index] = animation_values.get('iris_small', -0.2)
                pose[iris_small_index + 1] = animation_values.get('iris_small', -0.2)
        else:
            return
        
        # Apply common animation values
        if animation_values:
            # Set breathing
            breathing_index = self.get_parameter_index(PoseParameterCategory.BREATHING)
            if breathing_index >= 0:
                pose[breathing_index] = animation_values['breathing']
            
            # Set head rotation
            head_x_index = self.get_parameter_index(PoseParameterCategory.FACE_ROTATION, "head_x")
            head_y_index = self.get_parameter_index(PoseParameterCategory.FACE_ROTATION, "head_y")
            neck_z_index = self.get_parameter_index(PoseParameterCategory.FACE_ROTATION, "neck_z")
            if head_x_index >= 0:
                pose[head_x_index] = animation_values['head_x']
            if head_y_index >= 0:
                pose[head_y_index] = animation_values['head_y']
            if neck_z_index >= 0:
                pose[neck_z_index] = animation_values['neck_z']  # Map neck_z to neck_z
            
            # Set body rotation
            body_y_index = self.get_parameter_index(PoseParameterCategory.BODY_ROTATION, "body_y")
            body_z_index = self.get_parameter_index(PoseParameterCategory.BODY_ROTATION, "body_z")
            if body_y_index >= 0:
                pose[body_y_index] = animation_values['body_y']
            if body_z_index >= 0:
                pose[body_z_index] = animation_values['body_z']
            
            # Set blinking
            eye_index = self.get_parameter_index(PoseParameterCategory.EYE)
            if eye_index >= 0 and 'eye_wink' in animation_values:
                wink_value = animation_values['eye_wink']
                pose[eye_index] = wink_value
                pose[eye_index + 1] = wink_value
            
            # Set eye rotation
            iris_rotation_x_index = self.get_parameter_index(PoseParameterCategory.IRIS_ROTATION, "iris_rotation_x")
            iris_rotation_y_index = self.get_parameter_index(PoseParameterCategory.IRIS_ROTATION, "iris_rotation_y")
            if iris_rotation_x_index >= 0:
                pose[iris_rotation_x_index] = animation_values['iris_rotation_x']
            if iris_rotation_y_index >= 0:
                pose[iris_rotation_y_index] = animation_values['iris_rotation_y']
            
            # Map mouth shape parameters
            mouth_shapes = ['mouth_aaa', 'mouth_iii', 'mouth_uuu', 'mouth_eee', 'mouth_ooo']
            for shape in mouth_shapes:
                shape_index = self.get_parameter_index(PoseParameterCategory.MOUTH, shape)
                if shape_index >= 0:
                    pose[shape_index] = animation_values.get(shape, 0.0)
            
            # Set eye wink
            eye_wink_index = self.get_parameter_index(PoseParameterCategory.EYE, "eye_wink")
            if eye_wink_index >= 0 and 'eye_wink' in animation_values:
                wink_value = animation_values['eye_wink']
                pose[eye_wink_index] = wink_value
                pose[eye_wink_index + 1] = wink_value
        
        # Update image
        self.update_image(pose)

    def get_parameter_index(self, category: PoseParameterCategory, param_name: str = None) -> int:
        params = self.poser.get_pose_parameter_groups()
        for param in params:
            if param.get_category() == category:
                if param_name is None or param.get_group_name() == param_name:
                    return param.get_parameter_index()
        return -1

    def update_image(self, pose: List[float]):
        if self.torch_source_image is None:
            # Draw loading state
            with wx.MemoryDC(self.result_bitmap) as dc:
                dc.Clear()
                font = wx.Font(18, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
                dc.SetFont(font)
                w, h = dc.GetTextExtent("Loading...")
                dc.DrawText("Loading...", 
                           (self.IMAGE_SIZE - w) // 2, 
                           (self.IMAGE_SIZE - h) // 2)
            self.image_panel.Refresh()
            return
        
        pose_tensor = torch.tensor(pose, device=self.device)
        with torch.no_grad():
            output_image = self.poser.pose(
                self.torch_source_image, 
                pose_tensor, 
                0)[0].detach().cpu()
            
        numpy_image = convert_output_image_from_torch_to_numpy(output_image)
        
        # Create the bitmap only once if dimensions match
        if (not hasattr(self, 'result_bitmap') or 
            self.result_bitmap.GetWidth() != numpy_image.shape[0] or 
            self.result_bitmap.GetHeight() != numpy_image.shape[1]):
            self.result_bitmap = wx.Bitmap(numpy_image.shape[0], numpy_image.shape[1])
        
        # First create wx.Image from the numpy array
        wx_image = wx.Image(
            numpy_image.shape[0],
            numpy_image.shape[1],
            numpy_image[:, :, 0:3].tobytes(),
            numpy_image[:, :, 3].tobytes())
        
        # Then convert wx.Image to wx.Bitmap
        wx_bitmap = wx_image.ConvertToBitmap()
        
        # Use a with statement to ensure proper DC cleanup
        with wx.MemoryDC(self.result_bitmap) as dc:
            dc.Clear()
            dc.DrawBitmap(wx_bitmap, 0, 0)
        
        self.image_panel.Refresh()

    def paint_result(self, event):
        # Use BufferedPaintDC to prevent flickering
        with wx.BufferedPaintDC(self.image_panel) as dc:
            if hasattr(self, 'result_bitmap'):
                dc.Clear()
                # Center the bitmap if needed
                w, h = self.image_panel.GetSize()
                x = (w - self.result_bitmap.GetWidth()) // 2
                y = (h - self.result_bitmap.GetHeight()) // 2
                dc.DrawBitmap(self.result_bitmap, x, y)

    def on_close(self, event):
        # Properly stop the timer before closing
        if self.timer:
            self.timer.Stop()
        event.Skip()

if __name__ == "__main__":
    device = torch.device('cuda:0')
    app = wx.App()
    frame = AutonomousAnimationFrame(device)
    frame.Show()
    app.MainLoop() 