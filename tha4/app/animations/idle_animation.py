import math
import random
import time
from typing import Dict

from .base_animation import BaseAnimation


class IdleAnimation(BaseAnimation):
    def __init__(self):
        # Breathing
        self.breathing_cycle = 0.0  # 0.0 to 1.0
        self.breathing_speed = 0.5  # Full cycle per second
        
        # Blinking
        self.blink_cycle = 0.0  # 0.0 to 1.0
        self.blink_speed = 0.25  # Time for one blink
        self.time_until_next_blink = random.uniform(2.0, 4.0)
        
        # Enhanced head movement
        self.head_x_cycle = 0.0  # Side to side
        self.head_y_cycle = 0.0  # Up and down
        self.head_z_cycle = 0.0  # Tilt
        self.head_movement_speed = 0.15
        self.head_movement_amount = 0.4  # Increased from 0.3
        
        # Enhanced body movement
        self.body_y_cycle = 0.0  # Side to side
        self.body_z_cycle = 0.0  # Forward/backward
        self.body_movement_speed = 0.15  # Increased from 0.1 for more active movement
        self.body_movement_amount = 0.35  # Increased from 0.2 for larger movement range
        
        # Add slight body tilt for more natural sway
        self.body_tilt_cycle = 0.0
        self.body_tilt_speed = 0.15
        self.body_tilt_amount = 0.2
        
        # Add slight offset to make movement more interesting
        self.body_y_offset = random.uniform(-0.1, 0.1)
        self.body_z_offset = random.uniform(-0.1, 0.1)
        
        # Eye movement
        self.iris_x_cycle = 0.0  # Left-right eye movement
        self.iris_y_cycle = 0.0  # Up-down eye movement
        self.iris_movement_speed = 0.08  # Very slow eye movement
        self.iris_movement_amount = 0.4  # How far eyes move
        self.time_until_next_eye_movement = random.uniform(2.0, 4.0)
        self.current_eye_target = (0, 0)  # Current look target
        self.next_eye_target = (0, 0)     # Next look target
        self.eye_movement_progress = 1.0   # Progress to next target (0-1)
        
        # Look target system
        self.current_look_target = (0, 0)
        self.next_look_target = (0, 0)
        self.look_transition_progress = 1.0
        self.time_until_next_look = random.uniform(1.5, 3.0)
        self.look_change_duration = 1.0  # Slightly faster transitions
        
        # Head following parameters
        self.head_follow_amount = 0.8  # Increased head movement
        self.head_lag = 0.25  # Slightly quicker response
        
        # Interest point system - more varied looking points
        self.interest_points = [
            (-0.8, 0.3),     # Far upper left
            (0.8, 0.3),      # Far upper right
            (-0.8, 0),       # Far left
            (0.8, 0),        # Far right
            (0, 0.4),        # Straight up
            (-0.5, 0.35),    # Upper left
            (0.5, 0.35),     # Upper right
            (-0.5, -0.1),    # Lower left
            (0.5, -0.1),     # Lower right
            (0, -0.2),       # Down
            # Add some middle points
            (-0.4, 0.2),     # Mid upper left
            (0.4, 0.2),      # Mid upper right
            (-0.4, 0),       # Mid left
            (0.4, 0),        # Mid right
        ]
        
        # Point weights - make some positions more likely
        self.point_weights = [
            2.0,  # Far upper left
            2.0,  # Far upper right
            1.5,  # Far left
            1.5,  # Far right
            1.0,  # Straight up
            2.0,  # Upper left
            2.0,  # Upper right
            1.0,  # Lower left
            1.0,  # Lower right
            0.5,  # Down
            1.5,  # Mid upper left
            1.5,  # Mid upper right
            1.0,  # Mid left
            1.0,  # Mid right
        ]
        
        # Gentle sway
        self.sway_cycle = 0.0
        self.sway_speed = 0.2  # Very slow sway
        self.sway_amount = 0.05  # Subtle amount

    def update(self, delta_time: float) -> Dict[str, float]:
        """Update animation parameters"""
        # Update look target
        self.time_until_next_look -= delta_time
        if self.time_until_next_look <= 0:
            self.current_look_target = self.next_look_target
            
            if random.random() < 0.8:  # 80% chance to look at interesting point
                # Choose point based on weights
                total_weight = sum(self.point_weights)
                r = random.uniform(0, total_weight)
                cumulative_weight = 0
                chosen_index = 0
                
                for i, weight in enumerate(self.point_weights):
                    cumulative_weight += weight
                    if r <= cumulative_weight:
                        chosen_index = i
                        break
                
                self.next_look_target = self.interest_points[chosen_index]
                # Add subtle variation
                self.next_look_target = (
                    self.next_look_target[0] + random.uniform(-0.05, 0.05),
                    self.next_look_target[1] + random.uniform(-0.05, 0.05)
                )
            else:  # Return to neutral or slight offset
                self.next_look_target = (
                    random.uniform(-0.1, 0.1),  # Slight random neutral position
                    random.uniform(-0.05, 0.15)
                )
            
            self.look_transition_progress = 0.0
            # Vary the time between looks based on distance
            distance = math.sqrt(
                (self.next_look_target[0] - self.current_look_target[0]) ** 2 +
                (self.next_look_target[1] - self.current_look_target[1]) ** 2
            )
            # Longer pause for further movements
            self.time_until_next_look = random.uniform(1.5, 2.5) + distance * 0.5
        
        # Update look transition with smoother easing
        if self.look_transition_progress < 1.0:
            self.look_transition_progress = min(1.0, 
                self.look_transition_progress + delta_time / self.look_change_duration)
            
            # Smooth easing function (cubic)
            t = self.look_transition_progress
            t = t * t * (3 - 2 * t)  # Smoother curve
            
            # Calculate eye position
            iris_x = self.current_look_target[0] + (self.next_look_target[0] - self.current_look_target[0]) * t
            iris_y = self.current_look_target[1] + (self.next_look_target[1] - self.current_look_target[1]) * t
            
            # Head follows eyes with lag and smoother curve
            head_t = max(0, min(1, (self.look_transition_progress - self.head_lag) / (1 - self.head_lag)))
            head_t = head_t * head_t * (3 - 2 * head_t)  # Same smooth curve
            head_x = self.current_look_target[0] + (self.next_look_target[0] - self.current_look_target[0]) * head_t
            head_y = self.current_look_target[1] + (self.next_look_target[1] - self.current_look_target[1]) * head_t
        else:
            iris_x = self.next_look_target[0]
            iris_y = self.next_look_target[1]
            head_x = self.next_look_target[0]
            head_y = self.next_look_target[1]

        # Scale head movement
        head_x *= self.head_follow_amount
        head_y *= self.head_follow_amount

        # Update gentle sway
        self.sway_cycle = (self.sway_cycle + delta_time * self.sway_speed) % (math.pi * 2)
        sway = math.sin(self.sway_cycle) * self.sway_amount
        
        # Add very subtle micro-movements
        micro_movement = math.sin(time.time() * 3) * 0.01
        head_x += micro_movement + sway
        head_y += micro_movement * 0.5
        
        # Gentler head tilt
        head_z = math.sin(time.time() * 0.5) * 0.08  # Very gentle base tilt
        if self.look_transition_progress < 1.0:
            # Smoother tilt during movement
            tilt_amount = head_x * 0.15 * math.sin(self.look_transition_progress * math.pi)
            head_z += tilt_amount

        # Update breathing
        self.breathing_cycle = (self.breathing_cycle + delta_time * self.breathing_speed) % 1.0
        breathing = math.sin(self.breathing_cycle * math.pi * 2) * 0.5 + 0.5

        # Update blinking
        self.time_until_next_blink -= delta_time
        if self.time_until_next_blink <= 0:
            self.blink_cycle = 0.0
            self.time_until_next_blink = random.uniform(2.0, 4.0)
        
        if self.blink_cycle < 1.0:
            self.blink_cycle = min(1.0, self.blink_cycle + delta_time / self.blink_speed)
            blink = math.sin(self.blink_cycle * math.pi) if self.blink_cycle < 1.0 else 0.0
        else:
            blink = 0.0

        # Update eye movement
        self.time_until_next_eye_movement -= delta_time
        if self.time_until_next_eye_movement <= 0:
            # Set new eye target
            self.current_eye_target = self.next_eye_target
            self.next_eye_target = (
                random.uniform(-0.7, 0.7),  # x position
                random.uniform(-0.5, 0.3)   # y position - look down more than up
            )
            self.eye_movement_progress = 0.0
            self.time_until_next_eye_movement = random.uniform(2.0, 4.0)
        
        # Smoothly interpolate eye movement
        if self.eye_movement_progress < 1.0:
            self.eye_movement_progress = min(1.0, 
                self.eye_movement_progress + delta_time * 2.0)  # 0.5 seconds movement
            # Smooth easing function
            t = math.sin(self.eye_movement_progress * math.pi * 0.5)
            iris_x = self.current_eye_target[0] + (self.next_eye_target[0] - self.current_eye_target[0]) * t
            iris_y = self.current_eye_target[1] + (self.next_eye_target[1] - self.current_eye_target[1]) * t
        else:
            iris_x = self.next_eye_target[0]
            iris_y = self.next_eye_target[1]

        # Return animation parameters
        return {
            "head_x": head_x,
            "head_y": head_y,
            "head_z": head_z,
            "eye_blink": blink,
            "iris_x": iris_x,
            "iris_y": iris_y,
            "breathing": breathing,
            "body_y": math.sin(self.body_y_cycle) * self.body_movement_amount + self.body_y_offset,
            "body_z": math.sin(self.body_z_cycle) * self.body_movement_amount + self.body_z_offset,
            "body_tilt": math.sin(self.body_tilt_cycle) * self.body_tilt_amount
        } 