import math
import random
import time
from typing import Dict
from .base_animation import BaseAnimation

class SurpriseAnimation(BaseAnimation):
    def __init__(self):
        super().__init__()  # Initialize base animation (including blink)
        # Eye widening with more dramatic pulse
        self.eye_cycle = 0.0
        self.eye_speed = 0.8  # Increased from 0.4 for faster sparkle
        self.eye_amount = 0.9  # Increased from 0.6 for wider eyes
        
        # Eyebrow raising with gentle bounce
        self.brow_cycle = 0.0
        self.brow_speed = 0.4
        self.brow_amount = 0.5
        
        # Mouth movement
        self.mouth_cycle = 0.0
        self.mouth_speed = 0.4
        self.mouth_amount = 0.6
        self.mouth_quiver_speed = 1.5
        
        # Head movement - reduced speed
        self.head_cycle = 0.0
        self.head_speed = 1.2  # Reduced from 2.0
        self.head_amount = 0.3
        
        # Body bounce/sway - reduced speed
        self.bounce_cycle = 0.0
        self.bounce_speed = 1.8  # Reduced from 3.0
        self.bounce_amount = 0.2
        
        # Breathing
        self.breathing_cycle = 0.0
        self.breathing_speed = 1.0
        self.breathing_amount = 0.3
        
        # Add sparkle effect
        self.sparkle_cycle = 0.0
        self.sparkle_speed = 3.0
        self.sparkle_amount = 0.15
        
        # Iris parameters for sparkle - focus on size changes
        self.iris_cycle = 0.0
        self.iris_speed = 2.0  # Slightly slower pulsing
        self.iris_scale = -0.3  # Less dilated base size

    def update(self, delta_time: float) -> Dict[str, float]:
        # Get blink value
        blink = self.update_blink(delta_time)
        
        # Update all cycles
        self.bounce_cycle = (self.bounce_cycle + delta_time * self.bounce_speed) % (math.pi * 2)
        self.head_cycle = (self.head_cycle + delta_time * self.head_speed) % (math.pi * 2)
        self.eye_cycle = (self.eye_cycle + delta_time * self.eye_speed) % (math.pi * 2)
        self.brow_cycle = (self.brow_cycle + delta_time * self.brow_speed) % (math.pi * 2)
        self.mouth_cycle = (self.mouth_cycle + delta_time * self.mouth_speed) % (math.pi * 2)
        self.breathing_cycle = (self.breathing_cycle + delta_time * self.breathing_speed) % (math.pi * 2)
        self.sparkle_cycle = (self.sparkle_cycle + delta_time * self.sparkle_speed) % (math.pi * 2)
        self.iris_cycle = (self.iris_cycle + delta_time * self.iris_speed) % (math.pi * 2)
        
        # Calculate movements with more dynamic ranges
        bounce = math.sin(self.bounce_cycle) * self.bounce_amount
        head_x = (math.sin(self.head_cycle * 1.3) * self.head_amount) * 0.5
        head_y = abs(math.sin(self.head_cycle) * self.head_amount + bounce) * 0.5
        head_z = (math.sin(self.head_cycle * 0.7) * (self.head_amount * 0.5)) * 0.5
        
        # Micro-movements
        micro_time = time.time() * 5
        micro_movement = math.sin(micro_time) * 0.02
        head_x += micro_movement
        head_y += micro_movement * 0.5
        
        # Calculate sparkle effect
        sparkle = math.sin(self.sparkle_cycle) * self.sparkle_amount
        
        return {
            # Face rotation parameters
            "head_x": max(min(head_x, 0.3), -0.3),
            "head_y": max(min(head_y + micro_movement, 0.5), -0.2),
            "neck_z": max(min(head_z, 0.2), -0.2),
            
            # Eye parameters
            "eye_wink": blink,
            "eye_happy_wink": 0.0,
            "eye_surprised": 0.9 * (1.0 - blink),  # Reduce surprise during blink
            "eye_relaxed": 0.0,
            "eye_unimpressed": 0.0,
            "eye_raised_lower_eyelid": 0.2 * (1.0 - blink),
            
            # Eyebrow parameters
            "eyebrow_troubled": 0.0,
            "eyebrow_angry": 0.0,
            "eyebrow_lowered": 0.0,
            "eyebrow_raised": self.brow_amount + math.sin(self.brow_cycle) * 0.03,
            "eyebrow_happy": 0.0,
            "eyebrow_serious": 0.0,
            
            # Mouth parameters
            "mouth_aaa": 0.0,
            "mouth_iii": 0.0,
            "mouth_uuu": 0.0,
            "mouth_eee": 0.0,
            "mouth_ooo": self.mouth_amount + math.sin(self.mouth_cycle) * 0.02,
            "mouth_delta": math.sin(self.mouth_cycle * self.mouth_quiver_speed) * 0.02,
            "mouth_lowered_corner": 0.0,
            "mouth_raised_corner": 0.0,
            "mouth_smirk": 0.0,
            
            # Body rotation - more dynamic
            "body_y": bounce * 0.5,
            "body_z": -0.1 + bounce * 0.3,
            
            # Iris parameters
            "iris_small": self.iris_scale + abs(math.sin(self.iris_cycle)) * 0.08,  # Reduced pulsing amount
            "iris_rotation_x": 0.0,  # Keep eyes centered
            "iris_rotation_y": 0.0,  # Keep eyes centered
            
            # Breathing
            "breathing": math.sin(self.breathing_cycle) * self.breathing_amount + 0.3
        }
