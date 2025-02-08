import math
import random
from .base_animation import BaseAnimation

class HappyAnimation(BaseAnimation):
    def __init__(self):
        # Basic timing
        self.base_cycle = 0.0
        self.cycle_speed = 1.2  # Increased overall speed
        
        # Head movement - much more energetic, bouncy
        self.head_bounce_cycle = 0.0
        self.head_bounce_speed = 2.0      # Much faster bouncing
        self.head_bounce_amount = 0.6     # Much larger movement
        
        # Blinking - faster, more cheerful
        self.blink_cycle = 0.0
        self.blink_speed = 0.15           # Even quicker blinks
        self.time_until_next_blink = random.uniform(1.0, 2.0)  # More frequent blinking
        
        # Eye wideness - wider for more enthusiasm
        self.eye_wideness = 0.4           # Increased eye wideness
        
        # Mouth - very happy expression
        self.mouth_happy_value = 1.0      # Maximum smile
        self.mouth_cycle = 0.0
        self.mouth_variation_speed = 1.2   # Faster variation
        self.mouth_variation_amount = 0.2  # More variation
        
        # Add mouth openness for more expressive smile
        self.mouth_open_cycle = 0.0
        self.mouth_open_speed = 1.0       # Faster mouth movement
        self.mouth_open_amount = 0.4      # More open mouth
        
        # Body movement - very energetic
        self.body_bounce_cycle = 0.0
        self.body_bounce_speed = 1.8      # Much faster bouncing
        self.body_bounce_amount = 0.5     # Much larger movement
        
        # Head tilts - enthusiastic tilts
        self.head_tilt_cycle = 0.0
        self.head_tilt_speed = 0.8        # Faster tilts
        self.head_tilt_amount = 0.7       # Larger tilts
        
    def update(self, delta_time: float) -> dict:
        # Update cycles
        self.base_cycle = (self.base_cycle + delta_time * self.cycle_speed) % (math.pi * 2)
        self.head_bounce_cycle = (self.head_bounce_cycle + delta_time * self.head_bounce_speed) % (math.pi * 2)
        self.body_bounce_cycle = (self.body_bounce_cycle + delta_time * self.body_bounce_speed) % (math.pi * 2)
        self.head_tilt_cycle = (self.head_tilt_cycle + delta_time * self.head_tilt_speed) % (math.pi * 2)
        self.mouth_cycle = (self.mouth_cycle + delta_time * self.mouth_variation_speed) % (math.pi * 2)
        self.mouth_open_cycle = (self.mouth_open_cycle + delta_time * self.mouth_open_speed) % (math.pi * 2)
        
        # Update blinking
        self.time_until_next_blink -= delta_time
        if self.time_until_next_blink <= 0:
            self.blink_cycle = 0.0
            self.time_until_next_blink = random.uniform(1.0, 2.0)
        if self.blink_cycle < 1.0:
            self.blink_cycle = min(1.0, self.blink_cycle + delta_time / self.blink_speed)
            
        # Calculate movements with more energetic patterns
        # Very bouncy head movement
        head_y = math.sin(self.head_bounce_cycle) * self.head_bounce_amount
        # Add extra bounce using compound sine waves
        head_y += math.sin(self.head_bounce_cycle * 2) * self.head_bounce_amount * 0.3
        
        # More dynamic side-to-side movement
        head_x = math.sin(self.base_cycle * 0.7) * 0.4  # Wider side-to-side
        head_x += math.sin(self.base_cycle * 1.4) * 0.2  # Add faster secondary movement
        
        # More dynamic head tilt
        head_z = math.sin(self.head_tilt_cycle) * self.head_tilt_amount
        head_z += math.sin(self.head_tilt_cycle * 0.5) * self.head_tilt_amount * 0.3
        
        # Very bouncy body movement
        body_y = math.sin(self.body_bounce_cycle) * self.body_bounce_amount
        body_y += math.sin(self.body_bounce_cycle * 1.5) * self.body_bounce_amount * 0.3
        
        body_z = math.sin(self.base_cycle * 0.7) * 0.25
        body_z += math.sin(self.base_cycle * 1.4) * 0.15
        
        # More dynamic eye movement
        iris_x = math.sin(self.base_cycle * 0.4) * 0.3
        iris_x += math.sin(self.base_cycle * 0.8) * 0.15
        
        iris_y = math.sin(self.base_cycle * 0.5) * 0.25
        iris_y += math.sin(self.base_cycle * 1.0) * 0.1
        
        # Rest of the calculations...
        blink_value = math.sin(self.blink_cycle * math.pi) if self.blink_cycle < 1.0 else 0.0
        eye_openness = 1.0 - blink_value + self.eye_wideness
        
        # More dynamic mouth movement
        mouth_smile = self.mouth_happy_value + math.sin(self.mouth_cycle) * self.mouth_variation_amount
        mouth_open = math.sin(self.mouth_open_cycle) * self.mouth_open_amount
        mouth_open += math.sin(self.mouth_open_cycle * 2) * self.mouth_open_amount * 0.2
        
        # Enhanced happy face
        return {
            'breathing': 0.5 + math.sin(self.base_cycle) * 0.3,  # More pronounced breathing
            'head_x': head_x,
            'head_y': head_y,
            'head_z': head_z,
            'body_y': body_y,
            'body_z': body_z,
            'eye_openness': eye_openness,
            'mouth_happy': mouth_smile,
            'mouth_open': mouth_open,
            'mouth_form': 0.8,    # Wide smile
            'blink': blink_value,
            'iris_x': iris_x,
            'iris_y': iris_y,
            'mouth_smile': 1.0,  # Big smile
            'eye_squint': 0.3,    # Slight squint
            'eyebrow_raise': 0.4, # Slightly raised eyebrows
            'cheek_puff': 0.3,    # Puffed cheeks
        } 