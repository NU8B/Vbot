import math
import random
from .base_animation import BaseAnimation

class AngryAnimation(BaseAnimation):
    def __init__(self):
        # Basic timing
        self.base_cycle = 0.0
        self.cycle_speed = 0.8  # Faster movements
        
        # Head movement - tense
        self.head_tension_cycle = 0.0
        self.head_tension_speed = 0.6
        self.head_tension_amount = 0.3
        
        # Blinking - quick, intense
        self.blink_cycle = 0.0
        self.blink_speed = 0.2
        self.time_until_next_blink = random.uniform(1.5, 3.0)
        
        # Eye expression - squinted
        self.eye_squint = 0.5
        
        # Mouth - angry expression
        self.mouth_angry_value = 0.9
        self.mouth_cycle = 0.0
        self.mouth_variation_speed = 0.6
        self.mouth_variation_amount = 0.1
        
        # Body movement - tense
        self.body_tension_cycle = 0.0
        self.body_tension_speed = 0.5
        self.body_tension_amount = 0.3
        
    def update(self, delta_time: float) -> dict:
        # Update cycles
        self.base_cycle = (self.base_cycle + delta_time * self.cycle_speed) % (math.pi * 2)
        self.head_tension_cycle = (self.head_tension_cycle + delta_time * self.head_tension_speed) % (math.pi * 2)
        self.body_tension_cycle = (self.body_tension_cycle + delta_time * self.body_tension_speed) % (math.pi * 2)
        self.mouth_cycle = (self.mouth_cycle + delta_time * self.mouth_variation_speed) % (math.pi * 2)
        
        # Update blinking
        self.time_until_next_blink -= delta_time
        if self.time_until_next_blink <= 0:
            self.blink_cycle = 0.0
            self.time_until_next_blink = random.uniform(1.5, 3.0)
        
        if self.blink_cycle < 1.0:
            self.blink_cycle = min(1.0, self.blink_cycle + delta_time / self.blink_speed)
            
        # Calculate movements
        # Tense head movement
        head_y = math.sin(self.head_tension_cycle) * self.head_tension_amount
        head_x = math.sin(self.base_cycle * 0.5) * 0.2  # Side-to-side
        head_z = math.sin(self.head_tension_cycle * 0.5) * 0.3  # Tilt
        
        # Tense body movement
        body_y = math.sin(self.body_tension_cycle) * self.body_tension_amount
        body_z = math.sin(self.base_cycle * 0.4) * 0.2
        
        # Eye expression
        blink_value = math.sin(self.blink_cycle * math.pi) if self.blink_cycle < 1.0 else 0.0
        eye_openness = 1.0 - blink_value - self.eye_squint
        
        # Mouth variation
        mouth_value = self.mouth_angry_value + math.sin(self.mouth_cycle) * self.mouth_variation_amount
        
        # Enhanced angry face
        return {
            'breathing': 0.5 + math.sin(self.base_cycle) * 0.2,  # Tense breathing
            'head_x': head_x,
            'head_y': head_y,
            'head_z': head_z,
            'body_y': body_y,
            'body_z': body_z,
            'eye_openness': eye_openness,
            'mouth_angry': mouth_value,
            'blink': blink_value,
            'iris_x': math.sin(self.base_cycle * 0.2) * 0.2,
            'iris_y': 0.1,  # Looking slightly up
            'mouth_form': 0.6,    # Narrow mouth
            'eye_squint': 0.7,    # Strong squint
            'eyebrow_frown': 1.0, # Strong frown
            'eyebrow_angle': 0.3, # Angled eyebrows
            'nose_wrinkle': 0.5,  # Wrinkled nose
        } 