import numpy as np
from gymnasium import Env
from abc import ABC, abstractmethod
from .neural_nets import Critic, Policy
import torch

class Agent(ABC):
    @abstractmethod
    def act(self, obs: dict) -> int:
        """Subclasses must implement this method"""
        return NotImplemented

    def run(self, env: Env, seed = None):
        # Buffers
        obs_steps = []
        action_steps = []
        reward_steps = []
        terminated_steps = []
        truncated_steps = []
        # Start episode
        obs, info = env.reset(seed=seed)
        obs_steps.append(obs)
        while True:
            action = self.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            obs_steps.append(obs)
            action_steps.append(action)
            reward_steps.append(reward)
            terminated_steps.append(terminated)
            truncated_steps.append(truncated)
            if terminated or truncated:
                env.close()
                break
        return obs_steps, action_steps, reward_steps, terminated_steps, truncated_steps
    

class DRLAgent(Agent):
    def __init__(self, action_space, device: str = "cpu"):
        self.action_space = action_space
        num_classes = self.action_space.n
        self.policy = Policy(num_classes=num_classes, device=device)
        self.critic = Critic(device=device)

    def action_logits(self, obs: list[dict]):
        grid = np.stack([o['grid'] for o in obs])
        orientation = np.array([o['robot_orientation'] for o in obs])
        grid_tensor = torch.from_numpy(grid).float()
        orientation_tensor = torch.from_numpy(orientation).long()
        logits = self.policy(grid_tensor, orientation_tensor)
        return logits
    
    def values(self, obs: list[dict]):
        grid = np.stack([o['grid'] for o in obs])
        orientation = np.array([o['robot_orientation'] for o in obs])
        grid_tensor = torch.from_numpy(grid).float()
        orientation_tensor = torch.from_numpy(orientation).long()
        values = self.critic(grid_tensor, orientation_tensor) # (N, 1)
        values = values.squeeze(dim=-1)
        return values

    def act(self, obs: dict) -> int:
        with torch.inference_mode():
            logits = self.action_logits([obs])
        probs = torch.softmax(logits, dim=-1)
        action = int(torch.multinomial(probs, 1))
        return action


class RandomAgent(Agent):
    def act(self, obs: dict) -> int:
        return np.random.randint(0, 3, size=1).item()
        
