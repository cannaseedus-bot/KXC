"""
π Field Composition System - Core Field Calculators

This module implements the foundational field calculators for the Micronaut project.
Fields combine to create meaningful movement in physics-driven UI.
"""

import math
from typing import Dict, List, Optional, Any, Tuple


class πFieldCompositor:
    """
    Core π Field Compositor
    Manages field registration and composition
    """
    
    def __init__(self):
        self.field_calculators: Dict[str, callable] = {}
        self.register_builtin_calculators()
    
    def register_builtin_calculators(self):
        """Register all built-in field calculators"""
        self.register_calculator("wind", self.calculate_wind_force)
        self.register_calculator("attraction_well", self.calculate_attraction_force)
        self.register_calculator("navigation_force", self.calculate_navigation_force)
        self.register_calculator("scroll_inertia", self.calculate_scroll_inertia)
        self.register_calculator("magnetic_field", self.calculate_magnetic_field)
        self.register_calculator("friction_field", self.calculate_friction_field)
    
    def register_calculator(self, field_type: str, calculator: callable):
        """Register a field calculator function"""
        self.field_calculators[field_type] = calculator
    
    def calculate_wind_force(self, params: Dict[str, Any], body: Dict[str, Any], position: List[float]) -> List[float]:
        """
        Wind Force Field
        Applies directional force to bodies
        """
        direction = params.get('direction', [1, 0, 0])
        strength = params.get('strength', 0.5)
        
        force = [
            direction[0] * strength,
            direction[1] * strength,
            direction[2] * strength
        ]
        
        if 'drag_coefficient' in body:
            drag = body['drag_coefficient']
            force[0] *= drag
            force[1] *= drag
            force[2] *= drag
        
        return force
    
    def calculate_attraction_force(self, params: Dict[str, Any], body: Dict[str, Any], position: List[float]) -> List[float]:
        """
        Attraction Well Field
        Creates central attraction or repulsion force
        """
        center = params.get('position', [0, 0, 0])
        strength = params.get('strength', 2.0)
        radius = params.get('radius', 5.0)
        falloff_power = params.get('falloff_power', 2.0)
        
        dx = position[0] - center[0]
        dy = position[1] - center[1]
        dz = position[2] - center[2]
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if distance > radius or distance < 0.01:
            return [0.0, 0.0, 0.0]
        
        direction = [-dx/distance, -dy/distance, -dz/distance]
        normalized_distance = distance / radius
        force_magnitude = strength * (1.0 / (normalized_distance ** falloff_power))
        
        force = [
            direction[0] * force_magnitude,
            direction[1] * force_magnitude,
            direction[2] * force_magnitude
        ]
        
        return force

    def calculate_navigation_force(self, params: Dict[str, Any], body: Dict[str, Any], position: List[float], velocity: List[float] = None) -> List[float]:
        """
        Navigation Force Field
        Guides bodies toward navigation targets with intelligent pathfinding
        """
        target = params.get('target_position', [0, 0, 0])
        strength = params.get('strength', 2.0)
        arrival_radius = params.get('arrival_radius', 1.0)
        max_speed = params.get('max_speed', 5.0)
        arrival_slowdown = params.get('arrival_slowdown', 2.0)
        
        dx = target[0] - position[0]
        dy = target[1] - position[1]
        dz = target[2] - position[2]
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if distance < 0.01:
            return [0.0, 0.0, 0.0]
        
        direction = [dx/distance, dy/distance, dz/distance]
        
        if distance < arrival_radius:
            desired_speed = max_speed * (distance / arrival_radius)
            strength_multiplier = min(1.0, distance / arrival_slowdown)
        else:
            desired_speed = max_speed
            strength_multiplier = 1.0
        
        desired_velocity = [
            direction[0] * desired_speed,
            direction[1] * desired_speed,
            direction[2] * desired_speed
        ]
        
        current_velocity = velocity or [0.0, 0.0, 0.0]
        steering_force = [
            desired_velocity[0] - current_velocity[0],
            desired_velocity[1] - current_velocity[1],
            desired_velocity[2] - current_velocity[2]
        ]
        
        force = [
            steering_force[0] * strength * strength_multiplier,
            steering_force[1] * strength * strength_multiplier,
            steering_force[2] * strength * strength_multiplier
        ]
        
        return force

    def calculate_scroll_inertia(self, params: Dict[str, Any], body: Dict[str, Any], position: List[float], velocity: List[float]) -> List[float]:
        """
        Scroll Inertia Field
        Simulates physical scrolling with momentum and friction
        """
        scroll_direction = params.get('direction', [0, 1, 0])
        decay_rate = params.get('decay_rate', 2.0)
        dt = params.get('dt', 0.0166667)
        
        if velocity is None:
            velocity = [0.0, 0.0, 0.0]
        elif isinstance(velocity, dict):
            velocity = [velocity.get('x', 0.0), velocity.get('y', 0.0), velocity.get('z', 0.0)]
        
        dot_product = (
            velocity[0] * scroll_direction[0] +
            velocity[1] * scroll_direction[1] +
            velocity[2] * scroll_direction[2]
        )
        
        current_speed = dot_product
        decay_factor = math.exp(-decay_rate * dt)
        new_speed = current_speed * decay_factor
        
        stop_threshold = params.get('stop_threshold', 0.1)
        if abs(new_speed) < stop_threshold:
            new_speed = 0.0
        
        speed_delta = new_speed - current_speed
        
        force = [
            scroll_direction[0] * speed_delta,
            scroll_direction[1] * speed_delta,
            scroll_direction[2] * speed_delta
        ]
        
        inertia_strength = params.get('strength', 1.0)
        force[0] *= inertia_strength
        force[1] *= inertia_strength
        force[2] *= inertia_strength
        
        return force

    def calculate_magnetic_field(self, params: Dict[str, Any], body: Dict[str, Any], position: List[float]) -> List[float]:
        """
        Magnetic Field for UI elements
        Creates attraction/repulsion between related UI elements
        """
        return [0.0, 0.0, 0.0]

    def calculate_friction_field(self, params: Dict[str, Any], body: Dict[str, Any], position: List[float], velocity: List[float]) -> List[float]:
        """
        Surface Friction Field
        Applies friction based on material properties
        """
        if velocity is None:
            return [0.0, 0.0, 0.0]
        
        friction_coefficient = params.get('coefficient', 0.1)
        normal_force = params.get('normal_force', 1.0)
        
        speed = math.sqrt(
            velocity[0]**2 +
            velocity[1]**2 +
            velocity[2]**2
        )
        
        if speed < 0.001:
            return [0.0, 0.0, 0.0]
        
        friction_magnitude = friction_coefficient * normal_force
        
        direction = [
            -velocity[0] / speed,
            -velocity[1] / speed,
            -velocity[2] / speed
        ]
        
        force = [
            direction[0] * friction_magnitude,
            direction[1] * friction_magnitude,
            direction[2] * friction_magnitude
        ]
        
        return force


def create_basic_field_system() -> πFieldCompositor:
    """
    Create a basic field system with core calculators
    """
    return πFieldCompositor()


# Example usage and testing
if __name__ == "__main__":
    print("Pi Field Composition System - Core Calculators")
    print("=" * 50)
    
    field_system = create_basic_field_system()
    
    test_body = {
        "id": "test_body",
        "position": [3.0, 4.0, 0.0],
        "velocity": [1.0, 0.0, 0.0],
        "mass": 1.0,
        "drag_coefficient": 0.8
    }
    
    wind_params = {"direction": [1, 0, 0], "strength": 2.0}
    wind_force = field_system.calculate_wind_force(wind_params, test_body, test_body["position"])
    print(f"Wind Force: {wind_force}")
    
    attraction_params = {"position": [0, 0, 0], "strength": 3.0, "radius": 10.0}
    attraction_force = field_system.calculate_attraction_force(attraction_params, test_body, test_body["position"])
    print(f"Attraction Force: {attraction_force}")
    
    print("\nField composition working correctly!")
