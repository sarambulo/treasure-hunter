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

from environment.cell_states import encode_cell, WallObj, BarrelObj
from .grid import GridWorld
from .layouts import Layout
from .shapes import Cone
from .display import color_grid

class GridEnv(gym.Env):
    # Class attributes
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}
    actions = [
        {'label': 'up', 'displacement': np.array((-1, 0)), 'orientation': 0},
        {'label': 'left', 'displacement': np.array((0, -1)), 'orientation': 90},
        {'label': 'down', 'displacement': np.array((1, 0)), 'orientation': 180},
        {'label': 'right', 'displacement': np.array((0, 1)), 'orientation': 270},
    ]
    orientations = [0, 90, 180, 270]
    UNSEEN_CHANNEL = 0
    ROBOT_CHANNEL = 1
    INVALID_MOVE_PENALTY = -50

    def __init__(self, layout: Layout, render_mode=None):
        super(GridEnv, self).__init__()
        self.render_mode = render_mode
        # State
        self.robot_position_grid = np.array([0, 0])
        self.robot_orientation_degrees = 0
        self.world = GridWorld(
            layout.grid_cell_size, layout.grid_shape
        )
        self.world.add_objects(layout.objects)
        H, W = self.world.grid_shape
        self._observation_buffer = np.concat([
            np.zeros((2, H, W)), self.world.one_hot
        ], axis=0).astype(np.uint8) # (C: 2 + num_unique_object_types, H, W)
        self.map = GridWorld(
            self.world.cell_size, self.world.grid_shape, fill_value=encode_cell("unseen")
        )
        # Action and Observation spaces
        self.action_space = gym.spaces.Discrete(len(self.actions))
        observation_shape = self._observation_buffer.shape
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=observation_shape, dtype=np.uint8)

    def step(self, action: int):
        # Update the robot's state based on the action taken
        # Calculate the reward based on the new state
        # Return the new observation, reward, terminated flag, truncated flag, and info
        new_robot_position_grid = self._get_new_robot_position(action)
        new_robot_orientation_degrees = self._get_new_robot_orientation(action)
        final_robot_position_grid = self._move_robot(new_robot_position_grid, new_robot_orientation_degrees)
        if (final_robot_position_grid == new_robot_position_grid).all():
            cells_not_visited_before = (self.map.grid == encode_cell("unseen")).sum().item()
            self._reveal_fov()
            cells_not_visited_after = (self.map.grid == encode_cell("unseen")).sum().item()
            reward = cells_not_visited_before - cells_not_visited_after
        else:
            reward = self.INVALID_MOVE_PENALTY
        terminated = self.terminated
        truncated = False
        info = {} # Just to comply with gym.Env API
        new_observation = self._get_observation()
        return new_observation, float(reward), terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Reset the robot's state to the initial position
        # Return the initial observation
        self.map = GridWorld(
            self.world.cell_size, self.world.grid_shape, fill_value=encode_cell("unseen")
        )
        new_robot_orientation_degrees = self._random_robot_orientation()
        while True:
            new_robot_position_grid = self._random_robot_position()
            final_robot_position_grid = self._move_robot(new_robot_position_grid, new_robot_orientation_degrees)
            if (final_robot_position_grid == new_robot_position_grid).all():
                break
        self._reveal_fov()
        starting_observation = self._get_observation()
        info = {} # Just to comply with gym.Env API
        return starting_observation, info
    
    def render(self):
        """Render the environment for human viewing."""
        if self.render_mode == "rgb_array":
            masked_grid = np.where(
                self.map.grid == encode_cell("unseen"),
                encode_cell("unseen"),
                self.world.grid
            )
            row, column = self.robot_position_grid
            masked_grid[row, column] = encode_cell("robot")
            colored_grid = color_grid(masked_grid)
            return colored_grid
        
    def get_robot_position_grid(self):
        return self.robot_position_grid
    
    def get_robot_orientation_degrees(self):
        return self.robot_orientation_degrees
    
    def set_robot_position(self, x, y, orientation_degrees = None):
        row, column = self.world.coords_to_grid(x, y)
        if row is None or column is None:
            raise ValueError(f"Invalid (x, y) coords {(x, y)}")
        else:
            new_robot_position_grid = np.array([row, column])
            self._move_robot(new_robot_position_grid, orientation_degrees)
            
    
    @property
    def terminated(self):
        return (self.map.grid != encode_cell("unseen")).all()
    
    def _get_observation(self):
        unseen_cells = (self.map.grid == encode_cell("unseen")) # (H, W)
        self._observation_buffer[0] = unseen_cells
        # Robot channel is already handled by _move_robot
        # Mask object channels based on seen cells
        seen_mask = ~unseen_cells
        self._observation_buffer[2:] = seen_mask * self.world.one_hot # (H, W) * (C - 2, H, W) -> (C - 2, H, W)
        return self._observation_buffer
    
    def _move_robot(self, new_robot_position_grid: np.ndarray, new_robot_orientation_degrees: int | None = None) -> np.ndarray:
        """
            Side-effect: Moves the robot in the observation_buffer (inplace)
        """
        assert isinstance(new_robot_position_grid, np.ndarray)
        assert new_robot_position_grid.shape == (2,)
        assert (new_robot_position_grid < self.world.grid_shape).all()
        assert (new_robot_position_grid >= (0, 0)).all()
        # Check if the new position is empty
        row, column = new_robot_position_grid
        if self.world.grid[row, column] not in [encode_cell(WallObj), encode_cell(BarrelObj)]:
            # Move robot in observation buffer
            old_row, old_column = self.robot_position_grid
            self._observation_buffer[self.ROBOT_CHANNEL, old_row, old_column] = 0
            new_row, new_column = new_robot_position_grid
            self._observation_buffer[self.ROBOT_CHANNEL, new_row, new_column] = 1
            self.robot_position_grid = new_robot_position_grid
        if new_robot_orientation_degrees is not None:
            self.robot_orientation_degrees = new_robot_orientation_degrees
        return self.robot_position_grid
    
    def _get_new_robot_position(self, action: int):
        """
            Returns the new position of the robot based on the current position
            and the action taken.
        """
        assert self.action_space.contains(action), f"Unsupported action {action} for action space {self.action_space}"
        displacement = self.actions[action]['displacement']
        new_position = self.robot_position_grid + displacement
        # Clip values to be within the grid
        H, W = self.world.grid_shape
        new_position = np.clip(new_position, [0, 0], [H - 1, W - 1])
        return new_position
    
    def _random_robot_position(self) -> np.ndarray:
        shape = self.observation_space.shape
        assert shape is not None and len(shape) == 3
        C, H, W = shape
        new_position = self.np_random.integers(low=0, high=(H, W))
        return new_position
    
    def _get_new_robot_orientation(self, action: int) -> int:
        assert self.action_space.contains(action), f"Unsupported action {action} for action space {self.action_space}"
        return self.actions[action]['orientation']
    
    def _random_robot_orientation(self) -> int:
        orientation = self.np_random.choice(self.orientations, 1).item()
        return orientation
    
    def _reveal_fov(self):
        fov = self._get_fov(self.robot_position_grid, self.robot_orientation_degrees)
        self.map.add_cone(fov, 0)

    def _get_fov(self, robot_position_grid: np.ndarray, robot_orientation_degrees: int):
        x_grid, y_grid = robot_position_grid
        x_coord, y_coord = self.world.grid_to_coords(x_grid, y_grid)
        center = np.array([x_coord, y_coord])
        radius = 200
        angle_degrees = 60
        fov = Cone(
            center=center,
            radius=radius,
            angle_degrees=angle_degrees,
            orientation_degrees=robot_orientation_degrees
        )
        return fov
