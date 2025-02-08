import math
import random
from .base_animation import BaseAnimation

class SadAnimation(BaseAnimation):
    def __init__(self):
        # Basic timing
        self.base_cycle = 0.0
        self.cycle_speed = 0.4  # Slower movements
        
        # Head movement - downward tilt
        self.head_droop_cycle = 0.0
        self.head_droop_speed = 0.2
        self.head_droop_amount = 0.4
        self.constant_down_tilt = 0.3    # Reduced to avoid extreme tilt
        
        # Eyes - looking down
        self.eye_look_down = 0.3         # Changed to positive for looking down
        self.eye_squint = 0.5
        self.eye_base_openness = 0.6
        
        # Blinking - slower, longer blinks
        self.blink_cycle = 0.0
        self.blink_speed = 0.4           # Slower blinks
        self.time_until_next_blink = random.uniform(3.0, 5.0)
        self.long_blink_chance = 0.4     # Higher chance of long blinks
        
        # Mouth - sad expression
        self.mouth_sad_value = 0.8
        self.mouth_cycle = 0.0
        self.mouth_variation_speed = 0.3
        self.mouth_variation_amount = 0.1
        
        # Body movement - subtle, drooping
        self.body_droop_cycle = 0.0
        self.body_droop_speed = 0.2
        self.body_droop_amount = 0.2
        
    def update(self, delta_time: float) -> dict:
        # Update cycles
        self.base_cycle = (self.base_cycle + delta_time * self.cycle_speed) % (math.pi * 2)
        self.head_droop_cycle = (self.head_droop_cycle + delta_time * self.head_droop_speed) % (math.pi * 2)
        self.body_droop_cycle = (self.body_droop_cycle + delta_time * self.body_droop_speed) % (math.pi * 2)
        self.mouth_cycle = (self.mouth_cycle + delta_time * self.mouth_variation_speed) % (math.pi * 2)
        
        # Update blinking
        self.time_until_next_blink -= delta_time
        if self.time_until_next_blink <= 0:
            self.blink_cycle = 0.0
            if random.random() < self.long_blink_chance:
                self.blink_speed = 0.5  # Long blink
                self.time_until_next_blink = random.uniform(4.0, 6.0)
            else:
                self.blink_speed = 0.3  # Normal blink
                self.time_until_next_blink = random.uniform(3.0, 5.0)
        
        if self.blink_cycle < 1.0:
            self.blink_cycle = min(1.0, self.blink_cycle + delta_time / self.blink_speed)
            
        # Calculate movements
        # More pronounced downward head tilt
        head_y = math.sin(self.head_droop_cycle) * self.head_droop_amount  # Removed negative
        head_x = self.constant_down_tilt  # Use x rotation for looking down
        head_z = math.sin(self.head_droop_cycle * 0.5) * 0.2
        
        # Drooping body movement
        body_y = -math.sin(self.body_droop_cycle) * self.body_droop_amount
        body_z = math.sin(self.base_cycle * 0.2) * 0.1
        
        # Eye expression - more tired looking
        blink_value = math.sin(self.blink_cycle * math.pi) if self.blink_cycle < 1.0 else 0.0
        eye_openness = (self.eye_base_openness - blink_value - self.eye_squint) * 0.8  # Overall more closed
        
        # Mouth variation
        mouth_value = self.mouth_sad_value + math.sin(self.mouth_cycle) * self.mouth_variation_amount
        
        # Enhanced sad face
        return {
            'breathing': 0.5 + math.sin(self.base_cycle) * 0.1,  # Subtle breathing
            'head_x': head_x,            # This controls up/down head rotation
            'head_y': head_y,            # This controls left/right head rotation
            'head_z': head_z,            # This controls head tilt
            'body_y': body_y,
            'body_z': body_z,
            'eye_openness': eye_openness,
            'mouth_sad': 0.9,
            'mouth_form': 0.3,
            'eye_squint': 0.5,
            'eyebrow_frown': 0.8,
            'eyebrow_angle': -0.5,
            'nose_wrinkle': 0.2,
            'blink': blink_value,
            'iris_x': math.sin(self.base_cycle * 0.1) * 0.1,
            'iris_y': self.eye_look_down,  # Changed to positive for looking down
        } 