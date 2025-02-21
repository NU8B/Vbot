import math
import random
import time
from typing import Dict
from .base_animation import BaseAnimation

class HappyAnimation(BaseAnimation):
    def __init__(self):
        # Bouncy movement
        self.bounce_cycle = 0.0
        self.bounce_speed = 3.0
        self.bounce_amount = 0.2
        
        # Head movement
        self.head_cycle = 0.0
        self.head_speed = 2.0
        self.head_amount = 0.3
        
        # Eye sparkle
        self.eye_sparkle_cycle = 0.0
        self.eye_sparkle_speed = 4.0
        self.eye_sparkle_amount = 0.4
        
        # Mouth
        self.smile_amount = 0.8
        self.mouth_open_cycle = 0.0
        self.mouth_open_speed = 0.8
        self.mouth_open_amount = 0.8
        
        # Cheeks
        self.cheek_cycle = 0.0
        self.cheek_speed = 1.5
        self.cheek_amount = 0.6
        
        # Breathing (added)
        self.breathing_cycle = 0.0
        self.breathing_speed = 1.0
        self.breathing_amount = 0.3
        
        # Eyebrows (added)
        self.brow_cycle = 0.0
        self.brow_speed = 1.5
        self.brow_amount = 0.4
        
        # Add iris movement
        self.iris_cycle = 0.0
        self.iris_speed = 1.5
        self.iris_amount = 0.3

    def update(self, delta_time: float) -> Dict[str, float]:
        # Update all cycles
        self.bounce_cycle = (self.bounce_cycle + delta_time * self.bounce_speed) % (math.pi * 2)
        self.head_cycle = (self.head_cycle + delta_time * self.head_speed) % (math.pi * 2)
        self.eye_sparkle_cycle = (self.eye_sparkle_cycle + delta_time * self.eye_sparkle_speed) % (math.pi * 2)
        self.mouth_open_cycle = (self.mouth_open_cycle + delta_time * self.mouth_open_speed) % (math.pi * 2)
        self.cheek_cycle = (self.cheek_cycle + delta_time * self.cheek_speed) % (math.pi * 2)
        self.breathing_cycle = (self.breathing_cycle + delta_time * self.breathing_speed) % (math.pi * 2)
        self.brow_cycle = (self.brow_cycle + delta_time * self.brow_speed) % (math.pi * 2)
        self.iris_cycle = (self.iris_cycle + delta_time * self.iris_speed) % (math.pi * 2)
        
        # Calculate movements with correct ranges (-1.0 to 1.0)
        bounce = math.sin(self.bounce_cycle) * self.bounce_amount
        head_x = (math.sin(self.head_cycle * 1.3) * self.head_amount) * 0.5  # Scale to -0.5 to 0.5
        head_y = (math.sin(self.head_cycle) * self.head_amount + bounce) * 0.5  # Scale to -0.5 to 0.5
        head_z = (math.sin(self.head_cycle * 0.7) * (self.head_amount * 0.5)) * 0.5  # Scale to -0.5 to 0.5
        eye_sparkle = (math.sin(self.eye_sparkle_cycle) * 0.5 + 0.5) * self.eye_sparkle_amount
        mouth_open = (math.sin(self.mouth_open_cycle) * 0.5 + 0.5) * self.mouth_open_amount
        cheek = (math.sin(self.cheek_cycle) * 0.5 + 0.5) * self.cheek_amount
        breathing = math.sin(self.breathing_cycle) * self.breathing_amount
        brow_raise = (math.sin(self.brow_cycle) * 0.5 + 0.5) * self.brow_amount
        
        # Micro-movements
        micro_time = time.time() * 5
        micro_movement = math.sin(micro_time) * 0.02
        head_x += micro_movement
        head_y += micro_movement * 0.5
        
        # Add iris movement - bouncy for happy
        iris_x = math.sin(self.iris_cycle) * self.iris_amount + bounce * 0.1
        iris_y = math.cos(self.iris_cycle) * self.iris_amount + bounce * 0.15
        
        return {
            # Face rotation parameters (all within -1.0 to 1.0)
            "head_x": max(min(head_x, 1.0), -1.0),
            "head_y": max(min(head_y, 1.0), -1.0),
            "neck_z": max(min(head_z, 1.0), -1.0),
            
            # Eye parameters
            "eye_happy_wink": eye_sparkle,
            "eye_wink": 0.0,
            "eye_surprised": 0.0,
            "eye_relaxed": 0.3,
            "eye_unimpressed": 0.0,
            "eye_raised_lower_eyelid": 0.0,
            
            # Eyebrow parameters
            "eyebrow_happy": brow_raise,
            "eyebrow_troubled": 0.0,
            "eyebrow_angry": 0.0,
            "eyebrow_lowered": 0.0,
            "eyebrow_raised": 0.0,
            "eyebrow_serious": 0.0,
            
            # Mouth parameters
            "mouth_aaa": mouth_open,
            "mouth_iii": 0.0,
            "mouth_uuu": 0.0,
            "mouth_eee": 0.0,
            "mouth_ooo": 0.0,
            "mouth_delta": 0.0,
            "mouth_raised_corner": self.smile_amount,
            "mouth_lowered_corner": 0.0,
            "mouth_smirk": 0.0,
            
            # Body rotation
            "body_y": bounce * 0.7,
            "body_z": bounce * 0.5,
            
            # Iris parameters
            "iris_small": 0.0,
            "iris_rotation_x": iris_x,
            "iris_rotation_y": iris_y,
            
            # Breathing
            "breathing": breathing
        } 