import logging
import os
import sys
import time
import math  # Added for math functions
import random  # Added for random blinking intervals
from typing import List, Optional

import torch
import wx
import numpy as np  # Added for numpy array handling

from tha4.charmodel.character_model import CharacterModel
from tha4.poser.modes.mode_14 import create_poser
from tha4.poser.poser import PoseParameterCategory
from tha4.image_util import convert_output_image_from_torch_to_numpy  # Added for image conversion
from tha4.app.animations.happy_animation import HappyAnimation
from tha4.app.animations.sad_animation import SadAnimation
from tha4.app.animations.angry_animation import AngryAnimation

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
        self.head_z_cycle = 0.0  # Tilt
        self.head_movement_speed = 0.15
        self.head_movement_amount = 0.4  # Increased from 0.3
        
        # Enhanced body movement
        self.body_y_cycle = 0.0  # Side to side
        self.body_z_cycle = 0.0  # Forward/backward
        self.body_movement_speed = 0.15  # Increased from 0.1 for more active movement
        self.body_movement_amount = 0.35  # Increased from 0.2 for larger movement range
        
        # Add slight body tilt for more natural sway
        self.body_tilt_cycle = 0.0
        self.body_tilt_speed = 0.15
        self.body_tilt_amount = 0.2
        
        # Add slight offset to make movement more interesting
        self.body_y_offset = random.uniform(-0.1, 0.1)
        self.body_z_offset = random.uniform(-0.1, 0.1)
        
        # Eye movement
        self.iris_x_cycle = 0.0  # Left-right eye movement
        self.iris_y_cycle = 0.0  # Up-down eye movement
        self.iris_movement_speed = 0.08  # Very slow eye movement
        self.iris_movement_amount = 0.4  # How far eyes move
        self.time_until_next_eye_movement = random.uniform(2.0, 4.0)
        self.current_eye_target = (0, 0)  # Current look target
        self.next_eye_target = (0, 0)     # Next look target
        self.eye_movement_progress = 1.0   # Progress to next target (0-1)
        
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
        # Update look target
        self.time_until_next_look -= delta_time
        if self.time_until_next_look <= 0:
            self.current_look_target = self.next_look_target
            
            if random.random() < 0.8:  # 80% chance to look at interesting point
                # Choose point based on weights
                total_weight = sum(self.point_weights)
                r = random.uniform(0, total_weight)
                cumulative_weight = 0
                chosen_index = 0
                
                for i, weight in enumerate(self.point_weights):
                    cumulative_weight += weight
                    if r <= cumulative_weight:
                        chosen_index = i
                        break
                
                self.next_look_target = self.interest_points[chosen_index]
                # Add subtle variation
                self.next_look_target = (
                    self.next_look_target[0] + random.uniform(-0.05, 0.05),
                    self.next_look_target[1] + random.uniform(-0.05, 0.05)
                )
            else:  # Return to neutral or slight offset
                self.next_look_target = (
                    random.uniform(-0.1, 0.1),  # Slight random neutral position
                    random.uniform(-0.05, 0.15)
                )
            
            self.look_transition_progress = 0.0
            # Vary the time between looks based on distance
            distance = math.sqrt(
                (self.next_look_target[0] - self.current_look_target[0]) ** 2 +
                (self.next_look_target[1] - self.current_look_target[1]) ** 2
            )
            # Longer pause for further movements
            self.time_until_next_look = random.uniform(1.5, 2.5) + distance * 0.5
        
        # Update look transition with smoother easing
        if self.look_transition_progress < 1.0:
            self.look_transition_progress = min(1.0, 
                self.look_transition_progress + delta_time / self.look_change_duration)
            
            # Smooth easing function (cubic)
            t = self.look_transition_progress
            t = t * t * (3 - 2 * t)  # Smoother curve
            
            # Calculate eye position
            iris_x = self.current_look_target[0] + (self.next_look_target[0] - self.current_look_target[0]) * t
            iris_y = self.current_look_target[1] + (self.next_look_target[1] - self.current_look_target[1]) * t
            
            # Head follows eyes with lag and smoother curve
            head_t = max(0, min(1, (self.look_transition_progress - self.head_lag) / (1 - self.head_lag)))
            head_t = head_t * head_t * (3 - 2 * head_t)  # Same smooth curve
            head_x = self.current_look_target[0] + (self.next_look_target[0] - self.current_look_target[0]) * head_t
            head_y = self.current_look_target[1] + (self.next_look_target[1] - self.current_look_target[1]) * head_t
        else:
            iris_x = self.next_look_target[0]
            iris_y = self.next_look_target[1]
            head_x = self.next_look_target[0]
            head_y = self.next_look_target[1]

        # Scale head movement
        head_x *= self.head_follow_amount
        head_y *= self.head_follow_amount

        # Update gentle sway
        self.sway_cycle = (self.sway_cycle + delta_time * self.sway_speed) % (math.pi * 2)
        sway = math.sin(self.sway_cycle) * self.sway_amount
        
        # Add very subtle micro-movements
        micro_movement = math.sin(time.time() * 3) * 0.01
        head_x += micro_movement + sway
        head_y += micro_movement * 0.5
        
        # Gentler head tilt
        head_z = math.sin(time.time() * 0.5) * 0.08  # Very gentle base tilt
        if self.look_transition_progress < 1.0:
            # Smoother tilt during movement
            tilt_amount = head_x * 0.15 * math.sin(self.look_transition_progress * math.pi)
            head_z += tilt_amount

        # Update breathing
        self.breathing_cycle = (self.breathing_cycle + delta_time * self.breathing_speed) % 1.0

        # Update blinking
        self.time_until_next_blink -= delta_time
        if self.time_until_next_blink <= 0:
            self.blink_cycle = 0.0
            self.time_until_next_blink = random.uniform(2.0, 4.0)
        
        if self.blink_cycle < 1.0:
            self.blink_cycle = min(1.0, self.blink_cycle + delta_time / self.blink_speed)

        # Update eye movement
        self.time_until_next_eye_movement -= delta_time
        if self.time_until_next_eye_movement <= 0:
            # Set new eye target
            self.current_eye_target = self.next_eye_target
            self.next_eye_target = (
                random.uniform(-0.7, 0.7),  # x position
                random.uniform(-0.5, 0.3)   # y position - look down more than up
            )
            self.eye_movement_progress = 0.0
            self.time_until_next_eye_movement = random.uniform(2.0, 4.0)
        
        # Smoothly interpolate eye movement
        if self.eye_movement_progress < 1.0:
            self.eye_movement_progress = min(1.0, 
                self.eye_movement_progress + delta_time * 2.0)  # 0.5 seconds movement
            # Smooth easing function
            t = math.sin(self.eye_movement_progress * math.pi * 0.5)
            iris_x = self.current_eye_target[0] + (self.next_eye_target[0] - self.current_eye_target[0]) * t
            iris_y = self.current_eye_target[1] + (self.next_eye_target[1] - self.current_eye_target[1]) * t
        else:
            iris_x = self.next_eye_target[0]
            iris_y = self.next_eye_target[1]

        # Add micro-movements
        micro_movement = math.sin(time.time() * 10) * 0.02
        head_x += micro_movement
        head_y += micro_movement * 0.5
        
        # Add slight head tilt during movement
        head_z = math.sin(time.time() * 0.8) * 0.15  # Gentle head tilt
        if self.look_transition_progress < 1.0:
            # Add extra tilt during movement
            head_z += (head_x * 0.2) * math.sin(self.look_transition_progress * math.pi)

        # Update body movement cycles
        self.body_y_cycle = (self.body_y_cycle + delta_time * self.body_movement_speed) % (math.pi * 2)
        self.body_z_cycle = (self.body_z_cycle + delta_time * self.body_movement_speed * 0.7) % (math.pi * 2)
        self.body_tilt_cycle = (self.body_tilt_cycle + delta_time * self.body_tilt_speed) % (math.pi * 2)
        
        # Calculate body movement with more natural swaying
        body_y = (math.sin(self.body_y_cycle) * self.body_movement_amount) + self.body_y_offset
        body_z = (math.sin(self.body_z_cycle) * self.body_movement_amount * 0.6) + self.body_z_offset
        
        # Add slight tilt that follows the side-to-side movement
        body_tilt = math.sin(self.body_tilt_cycle) * self.body_tilt_amount
        # Add counter-tilt when moving sideways for more natural movement
        body_tilt += body_y * -0.15  # Counter-tilt against sideways movement
        
        # Add breathing influence to body movement
        breathing_influence = math.sin(self.breathing_cycle * math.pi * 2) * 0.1
        body_z += breathing_influence

        return {
            'breathing': math.sin(self.breathing_cycle * math.pi * 2) * 0.5 + 0.5,
            'head_x': head_x,
            'head_y': head_y,
            'head_z': head_z,
            'body_y': body_y,
            'body_z': body_z,
            'body_rotation_z': body_tilt,  # Add body tilt to rotation
            'blink': math.sin(self.blink_cycle * math.pi) if self.blink_cycle < 1.0 else 0.0,
            'iris_x': iris_x,
            'iris_y': iris_y,
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
            )
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
            # Apply sad-specific parameters
            mouth_sad_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_sad")
            if mouth_sad_index >= 0:
                pose[mouth_sad_index] = animation_values.get('mouth_sad', 0.8)
        elif self.animation_presets['angry'].is_active:
            animation_values = self.angry_animation.update(delta_time)
            # Apply angry-specific parameters
            mouth_angry_index = self.get_parameter_index(PoseParameterCategory.MOUTH, "mouth_angry")
            if mouth_angry_index >= 0:
                pose[mouth_angry_index] = animation_values.get('mouth_angry', 0.9)
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
            head_z_index = self.get_parameter_index(PoseParameterCategory.FACE_ROTATION, "neck_z")
            if head_x_index >= 0:
                pose[head_x_index] = animation_values['head_x']
            if head_y_index >= 0:
                pose[head_y_index] = animation_values['head_y']
            if head_z_index >= 0:
                pose[head_z_index] = animation_values['head_z']
            
            # Set body rotation
            body_y_index = self.get_parameter_index(PoseParameterCategory.BODY_ROTATION, "body_y")
            body_z_index = self.get_parameter_index(PoseParameterCategory.BODY_ROTATION, "body_z")
            if body_y_index >= 0:
                pose[body_y_index] = animation_values['body_y']
            if body_z_index >= 0:
                pose[body_z_index] = animation_values['body_z']
            
            # Set blinking
            eye_index = self.get_parameter_index(PoseParameterCategory.EYE)
            if eye_index >= 0 and 'blink' in animation_values:
                blink_value = animation_values['blink']
                pose[eye_index] = blink_value
                pose[eye_index + 1] = blink_value
            
            # Set eye rotation
            iris_x_index = self.get_parameter_index(PoseParameterCategory.IRIS_ROTATION, "iris_rotation_x")
            iris_y_index = self.get_parameter_index(PoseParameterCategory.IRIS_ROTATION, "iris_rotation_y")
            if iris_x_index >= 0:
                pose[iris_x_index] = animation_values['iris_x']
            if iris_y_index >= 0:
                pose[iris_y_index] = animation_values['iris_y']
        
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