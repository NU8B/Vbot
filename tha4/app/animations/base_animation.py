from abc import ABC, abstractmethod
from typing import Dict
import random
import math


class BaseAnimation(ABC):
    def __init__(self):
        # Common blinking parameters
        self.blink_cycle = 0.0
        self.blink_speed = 0.25  # Time for one blink
        self.time_until_next_blink = random.uniform(2.0, 4.0)
        
    def update_blink(self, delta_time: float) -> float:
        """Update blink animation and return blink value"""
        self.time_until_next_blink -= delta_time
        if self.time_until_next_blink <= 0:
            self.blink_cycle = 0.0
            self.time_until_next_blink = random.uniform(2.0, 4.0)
        
        if self.blink_cycle < 1.0:
            self.blink_cycle = min(1.0, self.blink_cycle + delta_time / self.blink_speed)
            
        return math.sin(self.blink_cycle * math.pi) if self.blink_cycle < 1.0 else 0.0

    @abstractmethod
    def update(self, delta_time: float) -> Dict[str, float]:
        """Update animation state and return animation parameters
        
        Args:
            delta_time (float): Time elapsed since last update in seconds
            
        Returns:
            Dict[str, float]: Dictionary mapping parameter names to values
        """
        pass

    def blend_with(self, other_animation: 'BaseAnimation', blend_factor: float) -> dict:
        """Blend between two animations"""
        pass 