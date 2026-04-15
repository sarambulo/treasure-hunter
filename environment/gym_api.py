"""
This module contains the Gym environment for the robot. It defines the action and observation spaces,
as well as the step and reset functions. The environment interacts with the 3D world map to update the
robot's state and calculate rewards based on its actions.

Design considerations:
- Skills: The robot should learn how to explore a room efficiently (look at each cell at least once and the mimimum number of times possible).
- Information: The robot needs to know its current position and orientation, the position of walls and doorways, and which cells have been visited.
- Actions: The robot will move in 8 possible directions (up, up-right, right, down-right, down, down-left, left, up-left).
- Success: Exploring the entire room with the minimum number of steps and without colliding with walls or doorways.
- End: The episode ends when the robot has explored all cells in the room or if the time runs out.

"""

import gymnasium as gym
import numpy as np
from typing import Optional

from environment.cell_states import encode_cell
from .grid import GridWorld
from .layout import Layout
from .shapes import Cone
from .display import color_grid

class GridEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 3}
    action_labels = [
        'forward',
        'turn-right',
        'turn-left',
        # 'right',
        # 'left',
        # 'backward'
    ]
    orientation_labels = [
        'up',
        'right',
        'down',
        'left',
        # 'up-right',
        # 'up-left'
        # 'down-right',
        # 'down-left',
    ]
    position_effects = {
        ('up', 'forward'): np.array((-1, 0)),
        ('right', 'forward'): np.array((0, 1)),
        ('down', 'forward'): np.array((1, 0)),
        ('left', 'forward'): np.array((0, -1)),
    }
    orientation_results = {
        ('up', 'turn-right'): 'right',
        ('up', 'turn-left'): 'left',
        ('right', 'turn-right'): 'down',
        ('right', 'turn-left'): 'up',
        ('down', 'turn-right'): 'left',
        ('down', 'turn-left'): 'right',
        ('left', 'turn-right'): 'up',
        ('left', 'turn-left'): 'down',
    }

    def __init__(self, layout: Layout, render_mode=None):
        super(GridEnv, self).__init__()
        self.render_mode = render_mode
        # State
        self.world = GridWorld(
            layout.grid_cell_size, layout.grid_bbox, layout.grid_shape, layout.grid_margin
        )
        self.world.add_objects(layout.objects)
        self.map = GridWorld(
            layout.grid_cell_size, layout.grid_bbox, layout.grid_shape, layout.grid_margin
        )
        self.map.grid = np.full_like(self.map.grid, encode_cell("unseen"))
        self.robot_position_grid = np.array([0, 0]) # in grid index coordinates
        self.robot_orientation = np.int64(0)
        # Action and Observation spaces
        self.action_space = gym.spaces.Discrete(len(self.action_labels))
        cell_values = self.world.possible_cell_values()
        space_shape = (len(cell_values), layout.grid_shape[0], layout.grid_shape[1])
        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.Box(low=0, high=1, shape=space_shape, dtype=np.uint8),
            "robot_orientation": gym.spaces.Discrete(len(self.orientation_labels)),
        })

    def step(self, action: int):
        # Update the robot's state based on the action taken
        # Calculate the reward based on the new state
        # Return the new observation, reward, terminated flag, truncated flag, and info
        cells_not_visited_before = (self.map.grid == encode_cell("unseen")).sum().item()
        self.robot_position_grid = self._new_robot_position(action)
        self.robot_orientation = self._new_robot_orientation(action)
        self._reveal_fov()
        cells_not_visited_after = (self.map.grid == encode_cell("unseen")).sum().item()
        reward = cells_not_visited_before - cells_not_visited_after
        terminated = self.terminated
        truncated = False
        info = {} # Just to comply with gym.Env API
        new_observation = self._get_observation()
        return new_observation, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Reset the robot's state to the initial position
        # Return the initial observation
        self.robot_position_grid = self._random_robot_position()
        self.robot_orientation = self._random_robot_orientation()
        self.map.grid = np.full_like(self.map.grid, encode_cell("unseen"))
        self._reveal_fov()
        starting_observation = self._get_observation()
        info = {} # Just to comply with gym.Env API
        return starting_observation, info
    
    def render(self):
        """Render the environment for human viewing."""
        if self.render_mode == "rgb_array":
            return color_grid(self._masked_world_map())
    
    @property
    def terminated(self):
        return (self.map.grid == 0).all()
    
    def _get_observation(self):
        # Get all posible combinations of row and column indices
        shape = self.observation_space["grid"].shape
        assert shape is not None and len(shape) == 3
        C, H, W = shape
        row_idx, col_idx = np.indices((H, W))
        # Initialize the output as all zeros
        one_hot = np.zeros((C, H, W), dtype=np.uint8)
        masked_grid = self._masked_world_map()
        zero_index_masked_grid = masked_grid - min(self.map.possible_cell_values()) # zero indexed
        one_hot[zero_index_masked_grid, row_idx, col_idx] = 1
        return {
            "grid": one_hot,
            "robot_orientation": self.robot_orientation
        }
    
    def _masked_world_map(self):
        # Use the seen_grid as a mask to decide which cells to
        # show the agent
        masked_grid = np.where(
            self.map.grid != encode_cell("unseen"),
            self.world.grid,
            self.map.grid
        )
        # Position the robot in the world map
        grid_x, grid_y = self.robot_position_grid
        masked_grid[grid_x, grid_y] = encode_cell("robot")
        return masked_grid
    
    def _new_robot_position(self, action: int):
        # Calculate the new position of the robot based on the action taken
        # This will depend on the current orientation and the action (e.g., move forward, turn right, etc.)
        assert self.action_space.contains(action), f"Unsupported action {action} for action space {self.action_space}"
        action_label = self.action_labels[action]
        orientation_label = self.orientation_labels[self.robot_orientation.item()]
        if (orientation_label, action_label) in self.position_effects:
            dxy = self.position_effects[(orientation_label, action_label)]
        else:
            dxy = np.array((0, 0))
        xy = self.robot_position_grid
        new_xy = xy + dxy
        # Clip values to be within the grid 
        shape = self.observation_space['grid'].shape
        assert shape is not None and len(shape) == 3
        C, H, W = shape
        new_xy = np.clip(new_xy, [0, 0], [H - 1, W - 1])
        return new_xy
    
    def _new_robot_orientation(self, action: int) -> np.int64:
        assert self.action_space.contains(action), f"Unsupported action {action} for action space {self.action_space}"
        current_orientation_label = self.orientation_labels[self.robot_orientation.item()]
        action_label = self.action_labels[action]
        if (current_orientation_label, action_label) in self.orientation_results:
            new_orientation_label = self.orientation_results[(current_orientation_label, action_label)]
            new_orientation = np.int64(self.orientation_labels.index(new_orientation_label))
        else:
            new_orientation = self.robot_orientation
        return new_orientation
    
    def _random_robot_position(self) -> np.ndarray:
        shape = self.observation_space['grid'].shape
        assert shape is not None and len(shape) == 3
        C, H, W = shape
        new_position = self.np_random.integers(low=0, high=(H, W))
        return new_position

    def _random_robot_orientation(self) -> np.int64:
        action_space_size = self.observation_space['robot_orientation'].n
        new_orientation = self.np_random.integers(0, action_space_size, 1)
        new_orientation = np.int64(new_orientation.item())
        return new_orientation
    
    def _reveal_fov(self):
        fov = self._get_fov()
        self.map.add_cone(fov, 0)

    def _get_fov(self):
        orientation_label = self.orientation_labels[self.robot_orientation.item()]
        x_grid, y_grid = self.robot_position_grid
        x_coord, y_coord = self.world.grid_to_coords(x_grid, y_grid)
        center = np.array([x_coord, y_coord])
        radius = 200
        angle_degrees = 60
        orientation_degrees = {
            "right": 0, "up": 90, "left": 180, "down": 270
        }[orientation_label] - angle_degrees/2
        fov = Cone(
            center=center,
            radius=radius,
            angle_degrees=angle_degrees,
            orientation_degrees=orientation_degrees
        )
        return fov
