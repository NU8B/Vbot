import math
import random
import time
from typing import Dict
from .base_animation import BaseAnimation

class SadAnimation(BaseAnimation):
    def __init__(self):
        # Droop movement
        self.droop_cycle = 0.0
        self.droop_speed = 0.5
        self.droop_amount = 0.4
        
        # Head movement
        self.head_cycle = 0.0
        self.head_speed = 0.3
        self.head_amount = 0.3
        
        # Eye movement
        self.eye_droop_cycle = 0.0
        self.eye_droop_speed = 0.4
        self.eye_droop_amount = 0.9
        
        # Mouth
        self.frown_amount = 0.7
        self.mouth_quiver_cycle = 0.0
        self.mouth_quiver_speed = 3.0
        self.mouth_quiver_amount = 0.1
        
        # Tears
        self.tear_cycle = 0.0
        self.tear_speed = 0.7
        self.tear_amount = 0.6
        
        # Breathing
        self.breathing_cycle = 0.0
        self.breathing_speed = 0.3
        self.breathing_amount = 0.4
        
        # Cheeks (added)
        self.cheek_cycle = 0.0
        self.cheek_speed = 0.4
        self.cheek_amount = 0.2
        
        # Eyebrows (added)
        self.brow_cycle = 0.0
        self.brow_speed = 0.5
        self.brow_amount = 0.6
        
        # Modify iris movement parameters
        self.iris_cycle = 0.0
        self.iris_speed = 0.0
        self.iris_amount = 0.0
        self.iris_down_offset = -0.6

    def update(self, delta_time: float) -> Dict[str, float]:
        # Update all cycles
        self.droop_cycle = (self.droop_cycle + delta_time * self.droop_speed) % (math.pi * 2)
        self.head_cycle = (self.head_cycle + delta_time * self.head_speed) % (math.pi * 2)
        self.eye_droop_cycle = (self.eye_droop_cycle + delta_time * self.eye_droop_speed) % (math.pi * 2)
        self.mouth_quiver_cycle = (self.mouth_quiver_cycle + delta_time * self.mouth_quiver_speed) % (math.pi * 2)
        self.tear_cycle = (self.tear_cycle + delta_time * self.tear_speed) % (math.pi * 2)
        self.breathing_cycle = (self.breathing_cycle + delta_time * self.breathing_speed) % (math.pi * 2)
        self.cheek_cycle = (self.cheek_cycle + delta_time * self.cheek_speed) % (math.pi * 2)
        self.brow_cycle = (self.brow_cycle + delta_time * self.brow_speed) % (math.pi * 2)
        self.iris_cycle = (self.iris_cycle + delta_time * self.iris_speed) % (math.pi * 2)
        
        # Calculate movements with correct ranges (-1.0 to 1.0)
        droop = math.sin(self.droop_cycle) * self.droop_amount
        head_x = (math.sin(self.head_cycle * 0.7) * self.head_amount - 0.5) * 0.5
        head_y = (math.sin(self.head_cycle) * self.head_amount + droop) * 0.5
        head_z = (math.sin(self.head_cycle * 0.5) * (self.head_amount * 0.3)) * 0.5
        eye_droop = (math.sin(self.eye_droop_cycle) * 0.5 + 0.5) * self.eye_droop_amount
        mouth_quiver = math.sin(self.mouth_quiver_cycle) * self.mouth_quiver_amount
        tear = (math.sin(self.tear_cycle) * 0.5 + 0.5) * self.tear_amount
        breathing = math.sin(self.breathing_cycle) * self.breathing_amount
        cheek = (math.sin(self.cheek_cycle) * 0.5 + 0.5) * self.cheek_amount
        brow_furrow = (math.sin(self.brow_cycle) * 0.5 + 0.5) * self.brow_amount
        
        # Micro-movements
        micro_time = time.time() * 2
        micro_movement = math.sin(micro_time) * 0.01
        head_x += micro_movement
        head_y += micro_movement * 0.3
        
        # Set iris values - looking downward
        iris_x = 0.0  # up down
        iris_y = 0.0 # left right
        
        return {
            # Face rotation parameters (all within -1.0 to 1.0)
            "head_x": max(min(head_x, 1.0), -1.0),
            "head_y": max(min(head_y, 0.5), -1.0),
            "neck_z": max(min(head_z, 1.0), -1.0),
            
            # Eye parameters
            "eye_wink": 0.0,
            "eye_happy_wink": 0.0,
            "eye_surprised": 0.0,
            "eye_relaxed": eye_droop * 1.0,
            "eye_unimpressed": 0.8,
            "eye_raised_lower_eyelid": 0.4,
            "eye_unimpressed_left": 0.8,
            "eye_unimpressed_right": 0.8,
            
            # Eyebrow parameters
            "eyebrow_troubled": brow_furrow,
            "eyebrow_angry": 0.0,
            "eyebrow_lowered": 0.0,
            "eyebrow_raised": 0.0,
            "eyebrow_happy": 0.0,
            "eyebrow_serious": 0.0,
            
            # Mouth parameters
            "mouth_aaa": 0.1,
            "mouth_iii": 0.0,
            "mouth_uuu": 0.0,
            "mouth_eee": 0.0,
            "mouth_ooo": 0.0,
            "mouth_delta": mouth_quiver,
            "mouth_lowered_corner": self.frown_amount,
            "mouth_raised_corner": 0.0,
            "mouth_smirk": 0.0,
            
            # Body rotation
            "body_y": droop * 0.5,
            "body_z": droop * 0.3,
            
            # Iris parameters
            "iris_small": 0.15,
            "iris_rotation_x": iris_x,
            "iris_rotation_y": iris_y,
            "iris_rotation_x_left": iris_x,
            "iris_rotation_y_left": iris_y,
            "iris_rotation_x_right": iris_x,
            "iris_rotation_y_right": iris_y,
            
            # Breathing
            "breathing": breathing
        } 