import math
import random
from .base_animation import BaseAnimation

class AngryAnimation(BaseAnimation):
    def __init__(self):
        # Basic timing
        self.base_cycle = 0.0
        self.cycle_speed = 0.8  # Medium speed for tense movements
        
        # Head movement - sharp, tense movements with downward tilt
        self.head_tense_cycle = 0.0
        self.head_tense_speed = 0.5
        self.head_tense_amount = 0.2
        self.head_forward_tilt = -0.1    # Downward tilt
        
        # Eyes - using exact parameters from image
        self.eye_unimpressed_value = 0.35  # Set to match image
        self.eye_base_openness = 0.8
        
        # Eyebrows - using exact angry parameter from image
        self.eyebrow_angry_value = 0.8  # Set to match image
        
        # Blinking - quick, intense blinks
        self.blink_cycle = 0.0
        self.blink_speed = 0.2           # Quick blinks
        self.time_until_next_blink = random.uniform(2.0, 4.0)
        
        # Enhanced mouth parameters for anger
        self.mouth_delta = 1.0          # Keep current delta for opening
        self.mouth_corner_droop_left = 1.0  # Maximum corner droop
        self.mouth_corner_droop_right = 1.0 # Maximum corner droop
        self.mouth_cycle = 0.0
        self.mouth_variation_speed = 0.4
        self.mouth_variation_amount = 0.05  # Small variation
        
        # Reset other mouth parameters to 0
        self.mouth_ooo_value = 0.0
        self.mouth_aaa_value = 0.0
        self.mouth_iii_value = 0.0
        self.mouth_uuu_value = 0.0
        self.mouth_eee_value = 0.0
        
        # Iris parameters
        self.iris_small_value = 0.2      # Smaller value = bigger iris
        
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
        # Head movements with downward tilt
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
            'breathing': 0.7 + math.sin(self.base_cycle) * 0.3,
            'head_x': head_x,
            'head_y': head_y,
            'head_z': head_z,
            'body_y': body_y,
            'body_z': body_z,
            'eye_openness': eye_openness,
            'eye_unimpressed_left': self.eye_unimpressed_value,
            'eye_unimpressed_right': self.eye_unimpressed_value,
            'eyebrow_angry_left': self.eyebrow_angry_value,
            'eyebrow_angry_right': self.eyebrow_angry_value,
            'mouth_delta': self.mouth_delta + mouth_variation,  # Only using delta
            'mouth_lowered_corner_left': self.mouth_corner_droop_left,
            'mouth_lowered_corner_right': self.mouth_corner_droop_right,
            'iris_small_left': self.iris_small_value,
            'iris_small_right': self.iris_small_value,
            'blink': blink_value,
            'iris_x': math.sin(self.base_cycle * 0.2) * 0.2,
            'iris_y': -0.1  # Slightly looking down
        } 