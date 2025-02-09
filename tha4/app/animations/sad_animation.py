import math
import random
import time
from typing import Dict
from .base_animation import BaseAnimation

class SadAnimation(BaseAnimation):
    def __init__(self):
        # Slow, heavy movement
        self.droop_cycle = 0.0
        self.droop_speed = 0.5  # Very slow
        self.droop_amount = 0.4  # Significant drooping
        
        # Head movement
        self.head_cycle = 0.0
        self.head_speed = 0.3  # Very slow head movement
        self.head_amount = 0.2  # Smaller movements
        
        # Eye movement
        self.eye_droop_cycle = 0.0
        self.eye_droop_speed = 0.4
        self.eye_droop_amount = 0.5
        
        # Mouth
        self.frown_amount = 0.7  # Deep frown
        self.mouth_quiver_cycle = 0.0
        self.mouth_quiver_speed = 3.0  # Fast quiver
        self.mouth_quiver_amount = 0.1  # Subtle quiver
        
        # Tears
        self.tear_cycle = 0.0
        self.tear_speed = 0.7
        self.tear_amount = 0.6
        
        # Breathing
        self.breathing_cycle = 0.0
        self.breathing_speed = 0.3  # Slow, heavy breathing
        self.breathing_amount = 0.4

    def update(self, delta_time: float) -> Dict[str, float]:
        # Update cycles
        self.droop_cycle = (self.droop_cycle + delta_time * self.droop_speed) % (math.pi * 2)
        self.head_cycle = (self.head_cycle + delta_time * self.head_speed) % (math.pi * 2)
        self.eye_droop_cycle = (self.eye_droop_cycle + delta_time * self.eye_droop_speed) % (math.pi * 2)
        self.mouth_quiver_cycle = (self.mouth_quiver_cycle + delta_time * self.mouth_quiver_speed) % (math.pi * 2)
        self.tear_cycle = (self.tear_cycle + delta_time * self.tear_speed) % (math.pi * 2)
        self.breathing_cycle = (self.breathing_cycle + delta_time * self.breathing_speed) % (math.pi * 2)
        
        # Calculate drooping movement
        droop = math.sin(self.droop_cycle) * self.droop_amount
        
        # Calculate head movement (slow and heavy)
        head_x = math.sin(self.head_cycle * 0.7) * self.head_amount
        head_y = math.sin(self.head_cycle) * self.head_amount + droop
        head_z = math.sin(self.head_cycle * 0.5) * (self.head_amount * 0.3)
        
        # Eye drooping effect
        eye_droop = (math.sin(self.eye_droop_cycle) * 0.5 + 0.5) * self.eye_droop_amount
        
        # Mouth quiver
        mouth_quiver = math.sin(self.mouth_quiver_cycle) * self.mouth_quiver_amount
        
        # Tear effect
        tear = (math.sin(self.tear_cycle) * 0.5 + 0.5) * self.tear_amount
        
        # Breathing
        breathing = math.sin(self.breathing_cycle) * self.breathing_amount
        
        # Add subtle micro-movements
        micro_time = time.time() * 2  # Slower micro-movements
        micro_movement = math.sin(micro_time) * 0.01  # Smaller movements
        head_x += micro_movement
        head_y += micro_movement * 0.3
        
        return {
            "head_x": head_x,
            "head_y": head_y,
            "head_z": head_z,
            "eye_droop": eye_droop,
            "mouth_frown": self.frown_amount + mouth_quiver,
            "tear": tear,
            "breathing": breathing,
            "body_y": droop * 0.5,
            "body_z": droop * 0.3,
            "eye_squint": 0.4  # Slightly squinted eyes
        } 