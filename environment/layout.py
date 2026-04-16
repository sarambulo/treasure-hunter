from dataclasses import dataclass, field
from aim_fsm import WallObj, BarrelObj, WallSpec, ArucoMarkerObj, Pose
import math
from typing import Sequence

@dataclass
class Layout:
    name: str = "Base"
    grid_cell_size: float = 10  # in mm
    grid_bbox=None
    grid_shape: tuple[int, int] = (40, 40)
    grid_margin: int = 0
    objects: Sequence[WallObj | BarrelObj] = field(default_factory=list)
    landmarks: dict[str, Pose] = field(default_factory=dict)

EmptyRoom = Layout(
    name = "EmptyRoom",
    objects = [
        WallObj(WallSpec({}, label="top_wall", length=800), x=400, y=0, z=0, theta=0),
        WallObj(WallSpec({}, label="bottom_wall", length=800), x=-400, y=0, z=0, theta=0),
        WallObj(WallSpec({}, label="right_wall", length=800), x=0, y=-400, z=0, theta=math.pi/2),
        WallObj(WallSpec({}, label="left_wall", length=800), x=0, y=400, z=0, theta=-math.pi/2),
    ],
    landmarks = {
        "Aruco_top_wall": Pose(400, 0, 0, math.pi),
        "Aruco_bottom_wall": Pose(-400, 0, 0, 0),
        "Aruco_right_wall": Pose(0, -400, 0, math.pi/2),
        "Aruco_left_wall": Pose(0, 400, 0, -math.pi/2),
    }
)
