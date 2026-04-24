from dataclasses import dataclass, field
from aim_fsm import WallObj, BarrelObj, WallSpec, ArucoMarkerObj, Pose
import math
from typing import Sequence

@dataclass
class Layout:
    name: str = "Base"
    grid_cell_size: float = 10  # in mm
    grid_shape: tuple[int, int] = (40, 40)
    objects: Sequence[WallObj | BarrelObj] = field(default_factory=list)
    landmarks: dict[str, Pose] = field(default_factory=dict)

EmptyRoom = Layout(
    name = "EmptyRoom",
    grid_shape = (80, 80),
    objects = [
        WallObj(WallSpec({}, label="top_wall", length=800), x=800, y=400, z=0, theta=math.pi),
        WallObj(WallSpec({}, label="bottom_wall", length=800), x=0, y=400, z=0, theta=0),
        WallObj(WallSpec({}, label="right_wall", length=800), x=400, y=0, z=0, theta=math.pi/2),
        WallObj(WallSpec({}, label="left_wall", length=800), x=400, y=800, z=0, theta=-math.pi/2),
    ],
    landmarks = {
        "ArucoMarker-1": Pose(800, 400, 0, math.pi),
        "ArucoMarker-2": Pose(400, 800, 0, -math.pi/2),
        "ArucoMarker-3": Pose(0, 400, 0, 0),
        "ArucoMarker-4": Pose(400, 0, 0, math.pi/2),
    }
)
