"""
Test script for field specification loading
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from field_composition_enhanced import πFieldSystem

def test_field_loading():
    """Test loading field specifications from JSON files"""
    print("Testing Field Specification Loading")
    print("=" * 40)
    
    # Create field system
    field_system = πFieldSystem()
    
    # Load specifications from specs directory
    specs_dir = os.path.join(os.path.dirname(__file__), "specs")
    
    print(f"Loading field specs from: {specs_dir}")
    
    # Load all field specs
    field_system.load_field_schema_directory(specs_dir)
    
    print(f"\nLoaded {len(field_system.fields)} field specifications:")
    
    for i, field in enumerate(field_system.fields):
        print(f"  {i+1}. {field['type']}: {field['description']}")
    
    # Test applying loaded fields to a body
    test_body = {
        "id": "test_body",
        "position": [3.0, 4.0, 0.0],
        "velocity": [1.0, 0.0, 0.0],
        "mass": 1.0,
        "type": "dynamic",
        "static": False
    }
    
    print(f"\nApplying fields to test body at position {test_body['position']}:")
    
    # Apply each field individually for testing
    for field in field_system.fields:
        field_type = field["type"]
        params = field["parameters"]
        
        # Get the appropriate calculator
        calculator = field_system.field_calculators.get(field_type)
        
        # Use the extracted parameters directly (they're already processed)
        if field_type == "wind":
            force = calculator(params, test_body, test_body["position"])
        elif field_type == "attraction_well":
            force = calculator(params, test_body, test_body["position"])
        elif field_type == "navigation_force":
            force = calculator(params, test_body, test_body["position"], test_body["velocity"])
        elif field_type == "scroll_inertia":
            force = calculator(params, test_body, test_body["position"], test_body["velocity"])
        else:
            force = calculator(params, test_body, test_body["position"])
        
        print(f"  {field_type}: force = [{force[0]:.3f}, {force[1]:.3f}, {force[2]:.3f}]")
    
    # Test applying all fields together
    total_force = field_system.apply_fields_to_body(test_body)
    print(f"\nTotal force from all fields: [{total_force[0]:.3f}, {total_force[1]:.3f}, {total_force[2]:.3f}]")
    
    print("\nField specification loading test completed successfully!")

if __name__ == "__main__":
    test_field_loading()