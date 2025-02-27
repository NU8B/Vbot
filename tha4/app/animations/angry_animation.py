import math
import random
import time
from typing import Dict
from .base_animation import BaseAnimation

class AngryAnimation(BaseAnimation):
    def __init__(self):
        super().__init__()  # Initialize base animation (including blink)
        # Tense, sharp movement
        self.tension_cycle = 0.0
        self.tension_speed = 2.0
        self.tension_amount = 0.3
        
        # Head movement
        self.head_cycle = 0.0
        self.head_speed = 1.5
        self.head_amount = 0.4
        
        # Eye movement
        self.eye_narrow_cycle = 0.0
        self.eye_narrow_speed = 0.8  # Slowed down for more stable narrow eyes
        self.eye_narrow_amount = 0.9  # Increased from 0.7 for more narrowed eyes
        
        # Mouth
        self.scowl_amount = 0.8  # Reduced from 1.0 to allow for natural frown
        self.mouth_tense_cycle = 0.0
        self.mouth_tense_speed = 2.0
        self.mouth_tense_amount = 0.2
        self.frown_amount = 1.0  # Added negative value for downward curve
        
        # Eyebrows
        self.brow_furrow_cycle = 0.0
        self.brow_furrow_speed = 1.2
        self.brow_furrow_amount = 1.0
        
        # Breathing
        self.breathing_cycle = 0.0
        self.breathing_speed = 1.5
        self.breathing_amount = 0.5
        
        # Cheeks (added for consistency)
        self.cheek_cycle = 0.0
        self.cheek_speed = 1.0
        self.cheek_amount = 0.2
        
        # Remove iris movement initialization since we want static iris

    def update(self, delta_time: float) -> Dict[str, float]:
        # Get blink value
        blink = self.update_blink(delta_time)
        
        # Update all cycles
        self.tension_cycle = (self.tension_cycle + delta_time * self.tension_speed) % (math.pi * 2)
        self.head_cycle = (self.head_cycle + delta_time * self.head_speed) % (math.pi * 2)
        self.eye_narrow_cycle = (self.eye_narrow_cycle + delta_time * self.eye_narrow_speed) % (math.pi * 2)
        self.mouth_tense_cycle = (self.mouth_tense_cycle + delta_time * self.mouth_tense_speed) % (math.pi * 2)
        self.brow_furrow_cycle = (self.brow_furrow_cycle + delta_time * self.brow_furrow_speed) % (math.pi * 2)
        self.breathing_cycle = (self.breathing_cycle + delta_time * self.breathing_speed) % (math.pi * 2)
        self.cheek_cycle = (self.cheek_cycle + delta_time * self.cheek_speed) % (math.pi * 2)
        
        # Calculate movements with correct ranges (-1.0 to 1.0)
        tension = math.sin(self.tension_cycle) * self.tension_amount
        head_x = (math.sin(self.head_cycle * 1.5) * self.head_amount) * 0.5  # Scale to -0.5 to 0.5
        head_y = (math.sin(self.head_cycle) * self.head_amount + tension) * 0.5  # Scale to -0.5 to 0.5
        head_z = (math.sin(self.head_cycle * 0.8) * (self.head_amount * 0.6)) * 0.5  # Scale to -0.5 to 0.5
        eye_narrow = (math.sin(self.eye_narrow_cycle) * 0.5 + 0.5) * self.eye_narrow_amount
        mouth_tense = math.sin(self.mouth_tense_cycle) * self.mouth_tense_amount
        brow_furrow = (math.sin(self.brow_furrow_cycle) * 0.5 + 0.5) * self.brow_furrow_amount
        breathing = math.sin(self.breathing_cycle) * self.breathing_amount
        cheek = (math.sin(self.cheek_cycle) * 0.5 + 0.5) * self.cheek_amount
        
        # Micro-movements
        micro_time = time.time() * 8
        micro_movement = math.sin(micro_time) * 0.03
        head_x += micro_movement
        head_y += micro_movement * 0.4
        
        # Remove iris movement calculation since we want static iris
        
        return {
            # Face rotation parameters (all within -1.0 to 1.0)
            "head_x": max(min(head_x, 1.0), -1.0),
            "head_y": max(min(head_y, 1.0), -1.0),
            "neck_z": max(min(head_z, 1.0), -1.0),
            
            # Eye parameters
            "eye_wink": blink,
            "eye_happy_wink": 0.0,
            "eye_surprised": 0.0,
            "eye_relaxed": 0.0,
            "eye_unimpressed": eye_narrow * (1.0 - blink),  # Reduce narrow eyes during blink
            "eye_raised_lower_eyelid": 0.4 * (1.0 - blink),
            
            # Eyebrow parameters
            "eyebrow_angry": brow_furrow,
            "eyebrow_troubled": 0.0,
            "eyebrow_lowered": 0.0,
            "eyebrow_raised": 0.0,
            "eyebrow_happy": 0.0,
            "eyebrow_serious": 0.0,
            
            # Mouth parameters
            "mouth_aaa": 0.0,
            "mouth_iii": 0.0,
            "mouth_uuu": 0.0,
            "mouth_eee": 0.0,
            "mouth_ooo": 0.0,
            "mouth_delta": mouth_tense + 0.5,
            "mouth_lowered_corner": self.scowl_amount,
            "mouth_raised_corner": 0.0,
            "mouth_smirk": 0.0,
            
            # Body rotation
            "body_y": tension * 0.6,
            "body_z": tension * 0.4,
            
            # Iris parameters - keep at default position and size
            "iris_small": 0.0,
            "iris_rotation_x": 0.0,
            "iris_rotation_y": 0.0,
            
            # Breathing
            "breathing": breathing
        } 