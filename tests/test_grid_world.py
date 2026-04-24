import pytest
from environment.grid import GridWorld
from aim_fsm import WallObj, WallSpec
from environment.cell_states import encode_cell
import matplotlib.pyplot as plt

def test_empty_world():
    grid_world = GridWorld()
    assert (grid_world.grid == encode_cell("empty")).all()

def test_index_to_coords():
    grid_world = GridWorld(square_size=10, grid_shape=(10, 10))
    assert grid_world.grid_to_coords(0, 0) == (95, 95)
    assert grid_world.grid_to_coords(1, 1) == (85, 85)
    assert grid_world.grid_to_coords(9, 9) == (5, 5)
    assert grid_world.grid_to_coords(10, 9) == (None, None)
    assert grid_world.grid_to_coords(-1, -1) == (None, None)

def test_coords_to_index():
    grid_world = GridWorld(square_size=10, grid_shape=(10, 10))
    assert grid_world.coords_to_grid(0, 0) == (9, 9)
    assert grid_world.coords_to_grid(12, 12) == (8, 8)
    assert grid_world.coords_to_grid(99.99, 99.99) == (0, 0)
    assert grid_world.coords_to_grid(100, 100) == (None, None)
    assert grid_world.coords_to_grid(-5, -5) == (None, None)

def test_add_obstacle():
    wall_object = WallObj(WallSpec({}, label="bottom_wall", length=100), x=0, y=50, z=0, theta=0)
    grid_world = GridWorld(square_size=10, grid_shape=(10, 10))
    grid_world.add_objects([wall_object])
    for y in range(100):
        row, column = grid_world.coords_to_grid(0, y)
        assert grid_world.grid[row, column] == encode_cell(WallObj)

def test_basic_room():
    from environment import EmptyRoom
    from environment.display import color_grid
    grid_world = GridWorld(
        square_size=EmptyRoom.grid_cell_size,
        grid_shape=EmptyRoom.grid_shape
    )
    grid_world.add_objects(EmptyRoom.objects)
    colored_grid = color_grid(grid_world.grid)
    plt.imshow(colored_grid)
    plt.savefig("tests/images/Basic room.jpeg")
    

