import math
import random
import time
from typing import Dict
from .base_animation import BaseAnimation

class AngryAnimation(BaseAnimation):
    def __init__(self):
        # Tense, sharp movement
        self.tension_cycle = 0.0
        self.tension_speed = 2.0  # Quick, sharp movements
        self.tension_amount = 0.3
        
        # Head movement
        self.head_cycle = 0.0
        self.head_speed = 1.5  # Quick head movements
        self.head_amount = 0.4  # Larger movements
        
        # Eye movement
        self.eye_narrow_cycle = 0.0
        self.eye_narrow_speed = 1.0
        self.eye_narrow_amount = 0.7  # Very narrowed eyes
        
        # Mouth
        self.scowl_amount = 0.8  # Deep scowl
        self.mouth_tense_cycle = 0.0
        self.mouth_tense_speed = 4.0  # Fast tension
        self.mouth_tense_amount = 0.2  # Visible tension
        
        # Eyebrows
        self.brow_furrow_cycle = 0.0
        self.brow_furrow_speed = 1.2
        self.brow_furrow_amount = 0.9  # Deep furrow
        
        # Breathing
        self.breathing_cycle = 0.0
        self.breathing_speed = 1.5  # Quick, agitated breathing
        self.breathing_amount = 0.5

    def update(self, delta_time: float) -> Dict[str, float]:
        # Update cycles
        self.tension_cycle = (self.tension_cycle + delta_time * self.tension_speed) % (math.pi * 2)
        self.head_cycle = (self.head_cycle + delta_time * self.head_speed) % (math.pi * 2)
        self.eye_narrow_cycle = (self.eye_narrow_cycle + delta_time * self.eye_narrow_speed) % (math.pi * 2)
        self.mouth_tense_cycle = (self.mouth_tense_cycle + delta_time * self.mouth_tense_speed) % (math.pi * 2)
        self.brow_furrow_cycle = (self.brow_furrow_cycle + delta_time * self.brow_furrow_speed) % (math.pi * 2)
        self.breathing_cycle = (self.breathing_cycle + delta_time * self.breathing_speed) % (math.pi * 2)
        
        # Calculate tense movement
        tension = math.sin(self.tension_cycle) * self.tension_amount
        
        # Calculate head movement (sharp and aggressive)
        head_x = math.sin(self.head_cycle * 1.5) * self.head_amount
        head_y = math.sin(self.head_cycle) * self.head_amount + tension
        head_z = math.sin(self.head_cycle * 0.8) * (self.head_amount * 0.6)
        
        # Eye narrowing effect
        eye_narrow = (math.sin(self.eye_narrow_cycle) * 0.5 + 0.5) * self.eye_narrow_amount
        
        # Mouth tension
        mouth_tense = math.sin(self.mouth_tense_cycle) * self.mouth_tense_amount
        
        # Brow furrow
        brow_furrow = (math.sin(self.brow_furrow_cycle) * 0.5 + 0.5) * self.brow_furrow_amount
        
        # Breathing
        breathing = math.sin(self.breathing_cycle) * self.breathing_amount
        
        # Add sharp micro-movements
        micro_time = time.time() * 8  # Fast micro-movements
        micro_movement = math.sin(micro_time) * 0.03  # Larger movements
        head_x += micro_movement
        head_y += micro_movement * 0.4
        
        return {
            "head_x": head_x,
            "head_y": head_y,
            "head_z": head_z,
            "eye_narrow": eye_narrow,
            "mouth_scowl": self.scowl_amount + mouth_tense,
            "brow_furrow": brow_furrow,
            "breathing": breathing,
            "body_y": tension * 0.6,
            "body_z": tension * 0.4,
            "eye_squint": 0.6  # Very squinted eyes
        } 