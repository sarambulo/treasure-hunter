import gymnasium as gym
import pytest
import environment
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

@pytest.fixture
def base_layout():
    return environment.Layout()

SEED = 50

def save_pic(env: gym.Env, filename: str):
    image = env.render()
    plt.imshow(image)
    plt.axis('off')
    file_path = Path('tests') / 'images' / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(file_path)

def test_init(base_layout):
    env = gym.make("GridEnv", layout=base_layout)

def test_reset(base_layout):
    env = gym.make("GridEnv", layout=base_layout, render_mode="rgb_array")
    obs, info = env.reset(seed=SEED)
    save_pic(env, f"Base Layout (seed {SEED}).png")
    assert (env.unwrapped.robot_position_grid == (31,31)).all()
    assert (env.unwrapped.robot_orientation == 2) # Down

def test_obs(base_layout):
    env = gym.make("GridEnv", layout=base_layout)
    obs, info = env.reset()
    assert isinstance(obs["grid"], np.ndarray)
    assert isinstance(obs["robot_orientation"].item(), int)

def test_actions(base_layout):
    env = gym.make("GridEnv", layout=base_layout, render_mode="rgb_array")
    obs, info = env.reset(seed=SEED) # Robot is as (31, 31) looking down
    new_obs, reward, terminated, truncated, info = env.step(2) # Turn left -> Now looking right
    new_obs, reward, terminated, truncated, info = env.step(0) # Move forward -> Moved right
    save_pic(env, f"Move right (seed {SEED})")
    ROBOT_CHANNEL = 2
    assert new_obs['grid'][ROBOT_CHANNEL, 31, 32] == 1
    UNSEEN_CHANNEL = 0
    assert new_obs['grid'][0, 31, 35] == 0

def test_fov(base_layout):
    env = gym.make("GridEnv", layout=base_layout)
    obs, info = env.reset(seed=SEED)
    UNSEEN_CHANNEL = 0
    # Space below should be seen now
    assert obs['grid'][UNSEEN_CHANNEL, 35, 31] == 0

def test_limits(base_layout):
    env = gym.make("GridEnv", layout=environment.Layout(grid_shape=(1, 1)))
    obs, info = env.reset(seed=SEED)
    ROBOT_CHANNEL = 2
    assert obs['grid'][ROBOT_CHANNEL, 0, 0] == 1
    assert int(obs['robot_orientation']) == 3 # Looking left
    # ====== TRY MOVING LEFT =========
    obs, reward, terminated, truncated, info = env.step(0) # Move left
    # Confirm no movement
    assert obs['grid'][ROBOT_CHANNEL, 0, 0] == 1
    assert int(obs['robot_orientation']) == 3 # Still looking left
    # ====== TRY MOVING UP =========
    env.step(1) # Turn righ, face up
    obs, reward, terminated, truncated, info = env.step(0) # Move up
    # Confirm no movement
    assert obs['grid'][ROBOT_CHANNEL, 0, 0] == 1
    assert int(obs['robot_orientation']) == 0 # Now looking up
    # ====== TRY MOVING RIGHT =========
    env.step(1) # Turn righ, face right
    obs, reward, terminated, truncated, info = env.step(0) # Move right
    # Confirm no movement
    assert obs['grid'][ROBOT_CHANNEL, 0, 0] == 1
    assert int(obs['robot_orientation']) == 1 # Now looking right
    # ====== TRY MOVING DOWN =========
    env.step(1) # Turn righ, face right
    obs, reward, terminated, truncated, info = env.step(0) # Move down
    # Confirm no movement
    assert obs['grid'][ROBOT_CHANNEL, 0, 0] == 1
    assert int(obs['robot_orientation']) == 2 # Now looking down