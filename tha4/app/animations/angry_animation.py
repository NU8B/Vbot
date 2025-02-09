import math
import random
from .base_animation import BaseAnimation

class AngryAnimation(BaseAnimation):
    def __init__(self):
        # Basic timing
        self.base_cycle = 0.0
        self.cycle_speed = 0.8  # Medium speed for tense movements
        
        # Head movement - sharp, tense movements
        self.head_tense_cycle = 0.0
        self.head_tense_speed = 0.6
        self.head_tense_amount = 0.3
        self.head_forward_tilt = 0.4    # Forward aggressive tilt
        
        # Eyes - using exact parameters from image
        self.eye_unimpressed_value = 0.35  # Set to match image
        self.eye_base_openness = 0.8
        
        # Eyebrows - using exact angry parameter from image
        self.eyebrow_angry_value = 0.8  # Set to match image
        
        # Blinking - quick, intense blinks
        self.blink_cycle = 0.0
        self.blink_speed = 0.2           # Quick blinks
        self.time_until_next_blink = random.uniform(2.0, 4.0)
        
        # Mouth parameters
        self.mouth_smirk_value = 1.0     # Maximum smirk for angry expression
        self.mouth_cycle = 0.0
        self.mouth_variation_speed = 0.4
        self.mouth_variation_amount = 0.1
        
        # Iris parameters
        self.iris_small_value = 0.2      # Smaller value = bigger iris (0 = biggest, 1 = smallest)
        
        # Body movement - tense, controlled
        self.body_tense_cycle = 0.0
        self.body_tense_speed = 0.4
        self.body_tense_amount = 0.2
        
    def update(self, delta_time: float) -> dict:
        # Update cycles
        self.base_cycle = (self.base_cycle + delta_time * self.cycle_speed) % (math.pi * 2)
        self.head_tense_cycle = (self.head_tense_cycle + delta_time * self.head_tense_speed) % (math.pi * 2)
        self.body_tense_cycle = (self.body_tense_cycle + delta_time * self.body_tense_speed) % (math.pi * 2)
        self.mouth_cycle = (self.mouth_cycle + delta_time * self.mouth_variation_speed) % (math.pi * 2)
        
        # Update blinking
        self.time_until_next_blink -= delta_time
        if self.time_until_next_blink <= 0:
            self.blink_cycle = 0.0
            self.time_until_next_blink = random.uniform(2.0, 4.0)
        
        if self.blink_cycle < 1.0:
            self.blink_cycle = min(1.0, self.blink_cycle + delta_time / self.blink_speed)
            
        # Calculate movements
        # Head movements
        head_x = math.sin(self.head_tense_cycle) * self.head_tense_amount + self.head_forward_tilt
        head_y = math.sin(self.base_cycle * 0.5) * 0.2
        head_z = math.sin(self.head_tense_cycle * 0.7) * 0.15
        
        # Tense body movement
        body_y = math.sin(self.body_tense_cycle) * self.body_tense_amount
        body_z = math.sin(self.base_cycle * 0.3) * 0.15
        
        # Eye expression
        blink_value = math.sin(self.blink_cycle * math.pi) if self.blink_cycle < 1.0 else 0.0
        eye_openness = (self.eye_base_openness - blink_value)
        
        # Calculate mouth variation
        mouth_variation = math.sin(self.mouth_cycle) * self.mouth_variation_amount
        
        # Return all parameters for angry expression
        return {
            'breathing': 0.7 + math.sin(self.base_cycle) * 0.3,  # More pronounced breathing
            'head_x': head_x,
            'head_y': head_y,
            'head_z': head_z,
            'body_y': body_y,
            'body_z': body_z,
            'eye_openness': eye_openness,
            'eye_unimpressed_left': self.eye_unimpressed_value,   # Exact from image
            'eye_unimpressed_right': self.eye_unimpressed_value,  # Exact from image
            'eyebrow_angry_left': self.eyebrow_angry_value,      # Exact from image
            'eyebrow_angry_right': self.eyebrow_angry_value,     # Exact from image
            'mouth_smirk': self.mouth_smirk_value + mouth_variation, # Maximum smirk with slight variation
            'iris_small_left': self.iris_small_value,            # Both eyes set to same value
            'iris_small_right': self.iris_small_value,           # Both eyes set to same value
            'blink': blink_value,
            'iris_x': math.sin(self.base_cycle * 0.2) * 0.2,
            'iris_y': -0.1  # Slightly looking down
        } 