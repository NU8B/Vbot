from abc import ABC, abstractmethod
from typing import Dict


class BaseAnimation(ABC):
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