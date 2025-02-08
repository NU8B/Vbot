from abc import ABC, abstractmethod

class BaseAnimation(ABC):
    @abstractmethod
    def update(self, delta_time: float) -> dict:
        """Update animation state and return parameter values"""
        pass

    def blend_with(self, other_animation: 'BaseAnimation', blend_factor: float) -> dict:
        """Blend between two animations"""
        pass 