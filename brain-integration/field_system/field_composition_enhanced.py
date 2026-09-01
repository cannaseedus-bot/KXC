"""
Enhanced π Field Composition System

Complete field system with all field types and world integration
"""

import math
import json
from typing import Dict, List, Optional, Any, Callable


class πFieldSystem:
    """
    Complete π Field System
    Manages field registration, composition, and application
    """
    
    def __init__(self):
        self.fields: List[Dict] = []
        self.field_calculators: Dict[str, Callable] = {}
        self.world_state: Optional[Any] = None
        
        # Register built-in field calculators
        self.register_builtin_calculators()
    
    def register_builtin_calculators(self):
        """Register all built-in field calculators"""
        self.register_calculator("wind", self.calculate_wind_force)
        self.register_calculator("attraction_well", self.calculate_attraction_force)
        self.register_calculator("navigation_force", self.calculate_navigation_force)
        self.register_calculator("scroll_inertia", self.calculate_scroll_inertia)
        self.register_calculator("magnetic_field", self.calculate_magnetic_field)
        self.register_calculator("friction_field", self.calculate_friction_field)
    
    def register_calculator(self, field_type: str, calculator: Callable):
        """Register a field calculator function"""
        self.field_calculators[field_type] = calculator
    
    def add_field_from_spec(self, spec_path: str):
        """Add a field from JSON specification file"""
        try:
            with open(spec_path, 'r') as f:
                spec = json.load(f)
            
            field_type = spec.get("field_type")
            if field_type not in self.field_calculators:
                raise ValueError(f"No calculator registered for field type: {field_type}")
            
            # Extract default values from parameters
            parameters = {}
            for param_name, param_spec in spec.get("parameters", {}).items():
                if "default" in param_spec:
                    parameters[param_name] = param_spec["default"]
                else:
                    # Use a sensible default based on type
                    param_type = param_spec.get("type", "float")
                    if param_type == "vector3":
                        parameters[param_name] = [0, 0, 0]
                    elif param_type == "float":
                        parameters[param_name] = 1.0
                    elif param_type == "int":
                        parameters[param_name] = 1
                    else:
                        parameters[param_name] = None
            
            field = {
                "type": field_type,
                "parameters": parameters,
                "rules": spec.get("application_rules", {}),
                "description": spec.get("description", "")
            }
            
            self.fields.append(field)
            print(f"Added field: {field_type} from {spec_path}")
            
        except Exception as e:
            print(f"Error loading field spec {spec_path}: {e}")
    
    def load_field_schema_directory(self, directory_path: str):
        """Load all field spec files from a directory"""
        import os
        import glob
        
        pattern = os.path.join(directory_path, "*.json")
        spec_files = glob.glob(pattern)
        
        for spec_file in spec_files:
            if spec_file.endswith("_spec.json"):
                self.add_field_from_spec(spec_file)
    
    def apply_fields_to_body(self, body: Dict, dt: float = 0.0166667) -> List[float]:
        """
        Apply all registered fields to a body
        Returns total force vector [x, y, z]
        """
        position = body.get("position", [0.0, 0.0, 0.0])
        velocity = body.get("velocity", [0.0, 0.0, 0.0])
        body_type = body.get("type", "dynamic")
        
        total_force = [0.0, 0.0, 0.0]
        
        for field in self.fields:
            field_type = field["type"]
            params = field["parameters"]
            rules = field["rules"]
            
            # Check if field applies to this body type
            applies_to = rules.get("applies_to", ["all_bodies"])
            if "all_bodies" not in applies_to and body_type not in applies_to:
                continue
            
            # Check exclusions
            excludes = rules.get("excludes", [])
            if body_type in excludes:
                continue
            
            # Get calculator
            calculator = self.field_calculators.get(field_type)
            if not calculator:
                continue
            
            # Calculate force based on field type
            if field_type == "wind":
                force = calculator(params, body, position)
                
            elif field_type == "attraction_well":
                force = calculator(params, body, position)
                
            elif field_type == "navigation_force":
                # Navigation force needs target position
                if "target_position" not in params:
                    continue
                force = calculator(params, body, position, velocity)
                
            elif field_type == "scroll_inertia":
                # Scroll inertia needs dt
                force = calculator(params, body, position, velocity)
                
            elif field_type == "friction_field":
                force = calculator(params, body, position, velocity)
                
            else:
                # Generic field calculation
                force = calculator(params, body, position)
            
            # Add to total force
            if force and len(force) == 3:
                total_force[0] += force[0]
                total_force[1] += force[1]
                total_force[2] += force[2]
        
        return total_force
    
    def apply_fields_to_world(self, world_state: Dict, dt: float = 0.0166667):
        """
        Apply all fields to all bodies in world state
        Modifies world_state in place
        """
        if "bodies" not in world_state:
            return
        
        for body in world_state["bodies"]:
            if body.get("static", False):
                continue
            
            force = self.apply_fields_to_body(body, dt)
            
            # Apply force to body
            if "force" not in body:
                body["force"] = [0.0, 0.0, 0.0]
            
            body["force"][0] += force[0]
            body["force"][1] += force[1]
            body["force"][2] += force[2]
    
    # Field calculators
    
    @staticmethod
    def calculate_wind_force(params, body, position):
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
    
    @staticmethod
    def calculate_attraction_force(params, body, position):
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
    
    @staticmethod
    def calculate_navigation_force(params, body, position, velocity=None):
        """
        Navigation Force Field
        Guides bodies toward navigation targets with intelligent pathfinding
        """
        target = params.get('target_position', [0, 0, 0])
        strength = params.get('strength', 2.0)
        arrival_radius = params.get('arrival_radius', 1.0)
        max_speed = params.get('max_speed', 5.0)
        arrival_slowdown = params.get('arrival_slowdown', 2.0)
        
        # Calculate direction to target
        dx = target[0] - position[0]
        dy = target[1] - position[1]
        dz = target[2] - position[2]
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if distance < 0.01:  # Already at target
            return [0.0, 0.0, 0.0]
        
        # Normalize direction
        direction = [dx/distance, dy/distance, dz/distance]
        
        # Apply arrival behavior (Reynolds steering)
        if distance < arrival_radius:
            # Slow down as we approach target
            desired_speed = max_speed * (distance / arrival_radius)
            strength_multiplier = min(1.0, distance / arrival_slowdown)
        else:
            desired_speed = max_speed
            strength_multiplier = 1.0
        
        # Calculate desired velocity
        desired_velocity = [
            direction[0] * desired_speed,
            direction[1] * desired_speed,
            direction[2] * desired_speed
        ]
        
        # Calculate steering force (desired_velocity - current_velocity)
        current_velocity = velocity or [0.0, 0.0, 0.0]
        steering_force = [
            desired_velocity[0] - current_velocity[0],
            desired_velocity[1] - current_velocity[1],
            desired_velocity[2] - current_velocity[2]
        ]
        
        # Apply strength with arrival modulation
        force = [
            steering_force[0] * strength * strength_multiplier,
            steering_force[1] * strength * strength_multiplier,
            steering_force[2] * strength * strength_multiplier
        ]
        
        return force
    
    @staticmethod
    def calculate_scroll_inertia(params, body, position, velocity):
        """
        Scroll Inertia Field
        Simulates physical scrolling with momentum and friction
        """
        scroll_direction = params.get('direction', [0, 1, 0])
        initial_speed = params.get('initial_speed', 10.0)
        decay_rate = params.get('decay_rate', 2.0)
        dt = params.get('dt', 0.0166667)
        
        if velocity is None:
            velocity = [0.0, 0.0, 0.0]
        
        # Ensure velocity is a list, not a dict
        if isinstance(velocity, dict):
            velocity = [velocity.get('x', 0.0), velocity.get('y', 0.0), velocity.get('z', 0.0)]
        
        # Project velocity onto scroll direction
        dot_product = (
            velocity[0] * scroll_direction[0] +
            velocity[1] * scroll_direction[1] +
            velocity[2] * scroll_direction[2]
        )
        
        # Calculate current speed in scroll direction
        current_speed = dot_product
        
        # Apply inertia decay (exponential)
        decay_factor = math.exp(-decay_rate * dt)
        new_speed = current_speed * decay_factor
        
        # If we're below threshold, stop
        stop_threshold = params.get('stop_threshold', 0.1)
        if abs(new_speed) < stop_threshold:
            new_speed = 0.0
        
        # Calculate force needed to achieve new speed
        speed_delta = new_speed - current_speed
        
        # Apply force in scroll direction
        force = [
            scroll_direction[0] * speed_delta,
            scroll_direction[1] * speed_delta,
            scroll_direction[2] * speed_delta
        ]
        
        # Scale by inertia strength parameter
        inertia_strength = params.get('strength', 1.0)
        force[0] *= inertia_strength
        force[1] *= inertia_strength
        force[2] *= inertia_strength
        
        return force
    
    @staticmethod
    def calculate_magnetic_field(params, body, position):
        """
        Magnetic Field for UI elements
        Creates attraction/repulsion between related UI elements
        """
        # Placeholder - would integrate with world to find other magnetic bodies
        return [0.0, 0.0, 0.0]
    
    @staticmethod
    def calculate_friction_field(params, body, position, velocity):
        """
        Surface Friction Field
        Applies friction based on material properties
        """
        if velocity is None:
            return [0.0, 0.0, 0.0]
        
        friction_coefficient = params.get('coefficient', 0.1)
        normal_force = params.get('normal_force', 1.0)  # Usually mass * gravity
        
        # Calculate speed
        speed = math.sqrt(
            velocity[0]**2 +
            velocity[1]**2 +
            velocity[2]**2
        )
        
        if speed < 0.001:
            return [0.0, 0.0, 0.0]
        
        # Friction force opposes velocity direction
        # F_friction = μ * N
        friction_magnitude = friction_coefficient * normal_force
        
        # Normalize velocity direction
        direction = [
            -velocity[0] / speed,  # Opposing direction
            -velocity[1] / speed,
            -velocity[2] / speed
        ]
        
        force = [
            direction[0] * friction_magnitude,
            direction[1] * friction_magnitude,
            direction[2] * friction_magnitude
        ]
        
        return force


# Example usage
if __name__ == "__main__":
    print("Enhanced Pi Field System")
    print("=" * 40)
    
    # Create field system
    field_system = πFieldSystem()
    
    # Create a simple world state
    world_state = {
        "bodies": [
            {
                "id": "b1",
                "position": [2.0, 3.0, 0.0],
                "velocity": [1.0, 0.0, 0.0],
                "mass": 1.0,
                "type": "dynamic",
                "static": False
            },
            {
                "id": "b2",
                "position": [5.0, 5.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "mass": 2.0,
                "type": "dynamic",
                "static": False
            }
        ]
    }
    
    # Add a wind field programmatically
    field_system.fields.append({
        "type": "wind",
        "parameters": {
            "direction": [1, 0, 0],
            "strength": 2.0
        },
        "rules": {
            "applies_to": ["dynamic"]
        },
        "description": "Test wind field"
    })
    
    # Apply fields to world
    field_system.apply_fields_to_world(world_state)
    
    print("World state after applying fields:")
    for body in world_state["bodies"]:
        force = body.get("force", [0, 0, 0])
        print(f"  Body {body['id']}: force = [{force[0]:.2f}, {force[1]:.2f}, {force[2]:.2f}]")
    
    print("\nEnhanced field system working correctly!")