import pytest
from environment import grid
from environment.grid import GridWorld
from aim_fsm import WorldObject, WallObj, BarrelObj, DoorwayObj
from aim_fsm.worldmap import WallSpec
from aim_fsm import geometry
from environment.cell_states import encode_cell

class MockWallSpec():
    def __init__(self, label = "mock_wall", length = 200, height = 100):
        self.label = label
        self.length = length
        self.height = height
        self.doorways = {}

@pytest.fixture
def wall_object():
    object = WallObj(x=50, y=50, z=0, theta=0, wall_spec=MockWallSpec())
    return object

def test_empty_world():
    grid_world = GridWorld()
    assert (grid_world.grid == encode_cell("empty")).all()

def test_index_to_coords():
    grid_world = GridWorld(square_size=10, bbox=((0,0),(100,100)))
    assert grid_world.grid_to_coords(0, 0) == (0, 0)
    assert grid_world.grid_to_coords(1, 1) == (10, 10)
    assert grid_world.grid_to_coords(5, 5) == (50, 50)

def test_coords_to_index():
    grid_world = GridWorld(square_size=10, bbox=((0,0),(100,100)))
    assert grid_world.coords_to_grid(0, 0) == (0, 0)
    assert grid_world.coords_to_grid(10, 10) == (1, 1)
    assert grid_world.coords_to_grid(50, 50) == (5, 5)
    assert grid_world.coords_to_grid(-5, -5) == (None, None)

def test_add_obstacle(wall_object):
    grid_world = GridWorld(square_size=10, bbox=((0,0),(100,100)))
    grid_world.add_objects([wall_object])
    for x in range(40, 60):
        assert grid_world.grid[grid_world.coords_to_grid(x,50)] == encode_cell(WallObj)
