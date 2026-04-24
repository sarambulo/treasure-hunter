import gymnasium as gym
from .api import GridEnv
from .layouts import *

# Register the environment so we can create it with gym.make()
gym.register(
    id="GridEnv",
    entry_point=GridEnv,
    max_episode_steps=300,  # Prevent infinite episodes
)