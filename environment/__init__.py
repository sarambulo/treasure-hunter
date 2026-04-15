import gymnasium as gym
from .gym_api import GridEnv
from .layout import Layout

# Register the environment so we can create it with gym.make()
gym.register(
    id="GridEnv",
    entry_point=GridEnv,
    max_episode_steps=300,  # Prevent infinite episodes
)