"""
This module has functions for extracting entities and their positions from the 3D world map.
For walls and barrels, it extracts their XYZ coordinates. For the robot, it extracts its XYZ
coordinates and orientation (yaw).
"""
from aim_fsm import WorldObject, WallObj, BarrelObj

RELEVANT_OBJECT_TYPES = [WallObj, BarrelObj]

def extract_entities(world_objects: list[WorldObject]):
    relevant_objects = [
        obj
        for obj in world_objects
        if isinstance(obj, tuple(RELEVANT_OBJECT_TYPES))
    ]
    extracted_entities = {
        
    }
    