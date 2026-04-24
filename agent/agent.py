import numpy as np
from gymnasium import Env
from abc import ABC, abstractmethod
from .neural_nets import Critic, Policy
from torch import nn
import torch

class Agent(ABC):
    @abstractmethod
    def act(self, obs: np.ndarray) -> int:
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
                break
        return obs_steps, action_steps, reward_steps, terminated_steps, truncated_steps
    

class DRLAgent(Agent):
    def __init__(self, action_space, device: str = "cpu"):
        self.action_space = action_space
        num_classes = self.action_space.n
        self.policy = Policy(num_classes=num_classes, device=device)
        self.critic = Critic(device=device)

    def action_logits(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.policy(obs) # (N, C)
        return logits
    
    def values(self, obs: torch.Tensor) -> torch.Tensor:
        values = self.critic(obs) # (N, 1)
        return values

    def act(self, obs: np.ndarray) -> int:
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        logits = self.action_logits(obs_tensor)
        probs = torch.softmax(logits, dim=-1)
        action = int(torch.multinomial(probs, 1))
        return action


class RandomAgent(Agent):
    def __init__(self, action_space):
        self.action_space = action_space
    def act(self, obs: np.ndarray) -> int:
        return self.action_space.sample()
        
class CartPoleAgent(Agent):
    def __init__(self, device: str = "cpu"):
        self.critic = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)
        self.policy = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        ).to(device)

    def action_logits(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.policy(obs) # (N, C)
        return logits
    
    def values(self, obs: torch.Tensor) -> torch.Tensor:
        values = self.critic(obs) # (N, 1)
        return values

    def act(self, obs: np.ndarray) -> int:
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        logits = self.action_logits(obs_tensor)
        probs = torch.softmax(logits, dim=-1)
        action = int(torch.multinomial(probs, 1))
        return action