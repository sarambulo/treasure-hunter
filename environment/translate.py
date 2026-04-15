"""
This module contains function for translating between the 3D world map and the
2D grid world environment.

The 3D world map is represented as a list of entities. Each entity has a type and position.
The 2D grid world environment is a tensor of dimensions (channels, height, width) where
each channel corresponds to a different type of entity (walls, barrels, robot). If the 
grid cell in that channel is occupied by that entity, the value is 1, otherwise it is 0.
"""

from .shapes import Circle, Rectangle
from aim_fsm import geometry, WallObj, BarrelObj, DoorwayObj
import numpy as np

# units: milimeters
BARREL_INFLATION_MM = 40
WALL_INFLATION_MM = 35
DOORWAY_INFLATION_MM = -62
DOORWAY_THICKNESS = 5

def wall_to_shape(wall):
    wall_spec = wall.wall_spec
    wall_half_length = wall_spec.length / 2
    widths = []
    edges = [ [0, -wall_half_length - WALL_INFLATION_MM, 0., 1.] ]
    last_x = -wall_half_length - WALL_INFLATION_MM
    for door_spec in wall_spec.doorways.values():
        door_center = door_spec['x']
        door_width = door_spec['width'] + DOORWAY_INFLATION_MM  # widen doorways for RRT, narrow for WaveFront
        left_edge = door_center - door_width/2 - wall_half_length
        edges.append([0., left_edge, 0., 1.])
        widths.append(left_edge - last_x)
        right_edge = door_center + door_width/2 - wall_half_length
        edges.append([0., right_edge, 0., 1.])
        last_x = right_edge
    edges.append([0., wall_half_length + WALL_INFLATION_MM, 0., 1.])
    widths.append(wall_half_length + WALL_INFLATION_MM - last_x)
    edges = np.array(edges).T
    edges = geometry.aboutZ(wall.pose.theta).dot(edges)
    edges = geometry.translate(wall.pose.x, wall.pose.y).dot(edges)
    obst = []
    for i in range(0,len(widths)):
        center = edges[:,2*i:2*i+2].mean(1).reshape(4,1)
        dimensions=(4.0+2*WALL_INFLATION_MM, widths[i])
        r = Rectangle(center=center,
                        dimensions=dimensions,
                        orient=wall.pose.theta )
        r.obstacle_id = wall.id
        obst.append(r)
    return obst

def barrel_to_shape(barrel):
    s = Circle(center=geometry.point(barrel.pose.x, barrel.pose.y),
                radius = barrel.diameter/2 + BARREL_INFLATION_MM)
    s.obstacle_id = barrel.id
    return s

def doorway_to_shape(doorway):
    s = Rectangle(center=geometry.point(doorway.pose.x, doorway.pose.y),
                    dimensions=[doorway.door_width+2*BARREL_INFLATION_MM, DOORWAY_THICKNESS+2*BARREL_INFLATION_MM],
                    orient=doorway.pose.theta)
    s.obstacle_id = doorway.id
    return s


OBJECT_TO_SHAPE_FUNCTIONS = {
    WallObj: wall_to_shape,
    BarrelObj: barrel_to_shape,
    DoorwayObj: doorway_to_shape,
}

def object_to_shape(obj):
    if type(obj) in OBJECT_TO_SHAPE_FUNCTIONS:
        return OBJECT_TO_SHAPE_FUNCTIONS[type(obj)](obj)
    else:
        raise ValueError("Unsupported object type: %s" % type(obj))