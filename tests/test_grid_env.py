import gymnasium as gym
import pytest
import environment
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from environment.api import GridEnv
from environment.layouts import EmptyRoom

@pytest.fixture
def base_layout():
    return environment.Layout()

SEED = 50

# ========================= UTILS ================================
def save_pic(env: gym.Env, filename: str):
    assert env.render_mode == "rgb_array"
    image = env.render()
    assert isinstance(image, np.ndarray)
    plt.imshow(image)
    plt.axis('off')
    file_path = Path('tests') / 'images' / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(file_path)

# ================================================================

def test_init(base_layout):
    env = gym.make("GridEnv", layout=base_layout)

def test_reset(base_layout):
    env = gym.make("GridEnv", layout=base_layout, render_mode="rgb_array")
    first_obs, info = env.reset(seed=SEED)
    second_obs, info = env.reset(seed=SEED)
    save_pic(env, f"Base Layout (seed {SEED}).png")
    assert (first_obs == second_obs).all()

def test_obs(base_layout):
    env = gym.make("GridEnv", layout=base_layout)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    row, column = env.unwrapped.get_robot_position_grid()
    robot_chanel = env.unwrapped.ROBOT_CHANNEL
    assert obs[robot_chanel, row, column] == 1

def test_actions(base_layout):
    env = gym.make("GridEnv", layout=base_layout, render_mode="rgb_array")
    obs, info = env.reset(seed=SEED)
    row, column = env.unwrapped.get_robot_position_grid()
    # Test moving up
    new_obs, reward, terminated, truncated, info = env.step(0) # Move up
    save_pic(env, f"Moved up.png")
    robot_channel = env.unwrapped.ROBOT_CHANNEL
    unseen_channel = env.unwrapped.UNSEEN_CHANNEL
    assert new_obs[robot_channel, row - 1 , column] == 1
    assert new_obs[unseen_channel, row - 5 , column] == 0
    # Test moving left
    row, column = env.unwrapped.get_robot_position_grid()   
    new_obs, reward, terminated, truncated, info = env.step(1) # Move left
    robot_channel = env.unwrapped.ROBOT_CHANNEL
    unseen_channel = env.unwrapped.UNSEEN_CHANNEL
    assert new_obs[robot_channel, row, column - 1] == 1
    assert new_obs[unseen_channel, row, column - 5] == 0

# def test_fov(base_layout):
#     env = gym.make("GridEnv", layout=base_layout)
#     obs, info = env.reset(seed=SEED)
#     UNSEEN_CHANNEL = 0
#     # Space below should be seen now
#     assert obs['grid'][UNSEEN_CHANNEL, 35, 31] == 0

def test_limits(base_layout):
    env = gym.make("GridEnv", layout=environment.Layout(grid_shape=(1, 1)))
    obs, info = env.reset(seed=SEED)
    ROBOT_CHANNEL = env.unwrapped.ROBOT_CHANNEL
    assert obs[ROBOT_CHANNEL, 0, 0] == 1
    # ====== TRY MOVING LEFT =========
    obs, reward, terminated, truncated, info = env.step(0) # Move up
    # Confirm no movement
    assert obs[ROBOT_CHANNEL, 0, 0] == 1
    # ====== TRY MOVING UP =========
    env.step(1) # Turn righ, face up
    obs, reward, terminated, truncated, info = env.step(1) # Move left
    # Confirm no movement
    assert obs[ROBOT_CHANNEL, 0, 0] == 1
    # ====== TRY MOVING RIGHT =========
    env.step(1) # Turn righ, face right
    obs, reward, terminated, truncated, info = env.step(2) # Move down
    # Confirm no movement
    assert obs[ROBOT_CHANNEL, 0, 0] == 1
    # ====== TRY MOVING DOWN =========
    env.step(1) # Turn righ, face right
    obs, reward, terminated, truncated, info = env.step(3) # Move right
    # Confirm no movement
    assert obs[ROBOT_CHANNEL, 0, 0] == 1

def test_walls():
    env = gym.make("GridEnv", layout=EmptyRoom, render_mode="rgb_array")
    env.reset(seed=SEED)
    save_pic(env, "Test Walls - Before moving.png")
    env.unwrapped.set_robot_position(49, 49) # Bottom right corner
    row, col = env.unwrapped.get_robot_position_grid()
    assert (row, col) == (75, 75)
    obs, reward, terminated, truncated, info = env.step(2) # Move down
    save_pic(env, "Test Walls - After moving.png")
    assert float(reward) < 0