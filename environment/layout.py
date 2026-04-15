from dataclasses import dataclass, field
from aim_fsm import WallObj, BarrelObj

@dataclass
class Layout:
    name: str = "Base"
    grid_cell_size: float = 10  # in mm
    grid_bbox=None
    grid_shape: tuple[int, int] = (40, 40)
    grid_margin: int = 0
    objects: list[WallObj | BarrelObj] = field(default_factory=list)


