"""
Test the π field system
"""

import sys
import os
import math

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from field_composition_enhanced import πFieldSystem

def test_basic_fields():
    """Test basic field composition"""
    print("Testing π Field System...")
    
    # Create field system
    field_system = πFieldSystem()
    
    # Load spec files from the specs directory
    specs_dir = os.path.join(os.path.dirname(__file__), "specs")
    field_system.load_field_schema_directory(specs_dir)
    
    # Test body
    test_body = {
        "id": "test_body",
        "position": [3.0, 4.0, 0.0],
        "velocity": [1.0, 0.0, 0.0],
        "mass": 1.0,
        "type": "dynamic",
        "static": False,
        "drag_coefficient": 0.5
    }
    
    # Test each field type
    print("\nTesting individual fields:")
    
    # Wind force
    wind_params = {"direction": [1, 0, 0], "strength": 2.0}
    wind_force = field_system.calculate_wind_force(wind_params, test_body, test_body["position"])
    print(f"Wind force: {wind_force}")
    
    # Attraction force
    attraction_params = {"position": [0, 0, 0], "strength": 3.0, "radius": 10.0}
    attraction_force = field_system.calculate_attraction_force(attraction_params, test_body, test_body["position"])
    print(f"Attraction force: {attraction_force}")
    
    # Navigation force
    nav_params = {"target_position": [10, 10, 0], "strength": 2.0, "arrival_radius": 1.0, "max_speed": 5.0, "arrival_slowdown": 2.0}
    nav_force = field_system.calculate_navigation_force(nav_params, test_body, test_body["position"], test_body["velocity"])
    print(f"Navigation force: {nav_force}")
    
    # Scroll inertia
    scroll_params = {"direction": [0, 1, 0], "initial_speed": 10.0, "decay_rate": 2.0, "dt": 0.0166667}
    scroll_force = field_system.calculate_scroll_inertia(scroll_params, test_body, test_body["position"], test_body["velocity"])
    print(f"Scroll inertia force: {scroll_force}")
    
    print("\nField system test complete!")

def test_world_simulation():
    """Test field application to a world"""
    print("\nTesting world simulation...")
    
    field_system = πFieldSystem()
    
    # Create a simple world
    world = {
        "gravity": [0, 9.81, 0],
        "bodies": [
            {
                "id": "body1",
                "position": [1.0, 2.0, 0.0],
                "velocity": [0.5, 0.0, 0.0],
                "mass": 1.0,
                "type": "dynamic",
                "static": False
            },
            {
                "id": "body2", 
                "position": [4.0, 1.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "mass": 2.0,
                "type": "static",
                "static": True
            }
        ]
    }
    
    # Add some fields
    field_system.fields.extend([
        {
            "type": "wind",
            "parameters": {"direction": [1, 0, 0], "strength": 1.5},
            "rules": {"applies_to": ["dynamic"]}
        },
        {
            "type": "attraction_well", 
            "parameters": {"position": [5, 5, 0], "strength": 0.5, "radius": 20},
            "rules": {"applies_to": ["dynamic"]}
        }
    ])
    
    # Apply fields
    field_system.apply_fields_to_world(world)
    
    print("World after field application:")
    for body in world["bodies"]:
        force = body.get("force", [0, 0, 0])
        print(f"  {body['id']}: force = [{force[0]:.2f}, {force[1]:.2f}, {force[2]:.2f}]")
    
    return world

if __name__ == "__main__":
    test_basic_fields()
    test_world_simulation()
