from aim_fsm import *
from environment import EmptyRoom

class Explore(StateMachineProgram):
    def __init__(self):
        for object in EmptyRoom.objects:
            if isinstance(object, WallObj):
                robot.world_map.objects[object.wall_spec.label] = object
        for i, (name, pose) in enumerate(EmptyRoom.landmarks.items()):
            landmark_object = ArucoMarkerObj(
                {'name': name, 'id': i, 'marker': None,},
                x=pose.x, y=pose.y, z=pose.z, theta=pose.theta,
                is_fixed=True
            )
            robot.world_map.objects[name] = landmark_object
        super().__init__()