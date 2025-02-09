import math
import random
import time
from typing import Dict
from .base_animation import BaseAnimation

class HappyAnimation(BaseAnimation):
    def __init__(self):
        # Bouncy movement
        self.bounce_cycle = 0.0
        self.bounce_speed = 3.0  # Faster for excitement
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
        self.smile_amount = 0.8  # Big smile
        self.mouth_open_cycle = 0.0
        self.mouth_open_speed = 2.0
        self.mouth_open_amount = 0.3
        
        # Cheeks
        self.cheek_cycle = 0.0
        self.cheek_speed = 1.5
        self.cheek_amount = 0.6

    def update(self, delta_time: float) -> Dict[str, float]:
        # Update cycles
        self.bounce_cycle = (self.bounce_cycle + delta_time * self.bounce_speed) % (math.pi * 2)
        self.head_cycle = (self.head_cycle + delta_time * self.head_speed) % (math.pi * 2)
        self.eye_sparkle_cycle = (self.eye_sparkle_cycle + delta_time * self.eye_sparkle_speed) % (math.pi * 2)
        self.mouth_open_cycle = (self.mouth_open_cycle + delta_time * self.mouth_open_speed) % (math.pi * 2)
        self.cheek_cycle = (self.cheek_cycle + delta_time * self.cheek_speed) % (math.pi * 2)
        
        # Calculate bouncy movement
        bounce = math.sin(self.bounce_cycle) * self.bounce_amount
        
        # Calculate head movement (more energetic)
        head_x = math.sin(self.head_cycle * 1.3) * self.head_amount
        head_y = math.sin(self.head_cycle) * self.head_amount + bounce
        head_z = math.sin(self.head_cycle * 0.7) * (self.head_amount * 0.5)
        
        # Eye sparkle effect
        eye_sparkle = (math.sin(self.eye_sparkle_cycle) * 0.5 + 0.5) * self.eye_sparkle_amount
        
        # Mouth movement
        mouth_open = (math.sin(self.mouth_open_cycle) * 0.5 + 0.5) * self.mouth_open_amount
        
        # Cheek movement (rosy cheeks)
        cheek = (math.sin(self.cheek_cycle) * 0.5 + 0.5) * self.cheek_amount
        
        # Add micro-movements for extra energy
        micro_time = time.time() * 5
        micro_movement = math.sin(micro_time) * 0.02
        head_x += micro_movement
        head_y += micro_movement * 0.5
        
        return {
            "head_x": head_x,
            "head_y": head_y,
            "head_z": head_z,
            "eye_sparkle": eye_sparkle,
            "mouth_smile": self.smile_amount,
            "mouth_open": mouth_open,
            "cheek": cheek,
            "body_y": bounce * 0.7,  # Slightly reduced body movement
            "body_z": bounce * 0.5,
            "eye_widen": 0.3  # Slightly wider eyes for excitement
        } 