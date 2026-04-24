import re

import gymnasium as gym
import yaml
from argparse import ArgumentParser
import numpy as np
import torch
from tqdm import tqdm
from dotenv import load_dotenv
import wandb
from pathlib import Path

# Local modules
import environment
from environment import Layout, EmptyRoom
from agent import RandomAgent, DRLAgent, CartPoleAgent


def main():
    cli_args = parse_cli_args()
    config = parse_config(cli_args.config)
    device = cli_args.device
    load_dotenv()
    run = wandb.init(
        entity="sarambulo-carnegie-mellon-university",
        project="cogrob",
        config=config,
        id=cli_args.id,
        resume="must" if cli_args.id else "auto",
        mode="disabled" if cli_args.disable_wandb else "online"
    )

    # Environment setup
    if config["environment"] == "GridEnv":
        env = gym.make(
            config["environment"],
            max_episode_steps=config["max_episode_steps"],
            layout=EmptyRoom,
            render_mode="rgb_array",
        )
    elif config["environment"] == "CartPole-v1":
        env = gym.make(
            config["environment"],
            max_episode_steps=config["max_episode_steps"],
            render_mode="rgb_array",
        )
    else:
        raise ValueError(f"Unsupported environment: {config['environment']}")
    record_episode = lambda x: (x + 1) % config["record_every"] == 0
    env = gym.wrappers.RecordVideo(
        env, "runs",
        episode_trigger=record_episode, name_prefix=f"train-{run.name}")
    # Agent setup
    if config['environment'] == "GridEnv":
        agent = DRLAgent(action_space=env.action_space, device=device)
    elif config['environment'] == 'CartPole-v1':
        agent = CartPoleAgent(device=device)
    else:
        raise ValueError(f"Unsupported environment: {config['environment']}")
    # Learning setup
    policy_optimizer = torch.optim.AdamW(agent.policy.parameters(), lr=config["policy_lr"])
    critic_optimizer = torch.optim.AdamW(agent.critic.parameters(), lr=config["critic_lr"])
    policy_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        policy_optimizer, gamma=config["policy_lr_decay"]
    )
    critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        critic_optimizer, gamma=config["critic_lr_decay"]
    )
    # Setup checkpoint dir
    checkpoint_dir = Path(cli_args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    # Load checkpoint if resuming a run
    if cli_args.id:
        first_episode = load_checkpoint(
            str(checkpoint_dir / f"{run.name}-last.pt"),
            agent, policy_optimizer, critic_optimizer
        )
    elif cli_args.pretrained:
        load_checkpoint(
            cli_args.pretrained,
            agent, policy_optimizer, critic_optimizer
        )
        first_episode = 0
    else:
        first_episode = 0
    last_episode = first_episode + config["train_episodes"]
    episode_progress_bar = tqdm(range(first_episode, last_episode), desc="Training episodes")
    for episode_number in episode_progress_bar:
        episode_steps = agent.run(env)
        observations, actions, rewards, terminated, truncated = collate_episode(episode_steps)
        values = agent.values(observations)
        returns = get_n_step_returns(
            rewards, values, terminated, discount_factor=config["discount_factor"], n=config["n_step"]
        ).to(device)
        logits = agent.action_logits(observations[:-1])
        assert logits.size(0) == actions.size(0)
        actions = actions.to(logits.device)
        policy_log_probs = torch.log_softmax(logits, dim=-1).gather(dim=-1, index=actions)
        advantages = returns - values[:-1]
        policy_loss = (-advantages.detach() * policy_log_probs).mean()
        critic_target = (rewards + config["discount_factor"] * values[1:]).detach()
        critic_loss = torch.nn.functional.mse_loss(values[:-1], critic_target)
        policy_loss = policy_loss / config["acum_gradients"]
        critic_loss = critic_loss / config["acum_gradients"]
        policy_loss.backward()
        critic_loss.backward()
        if (episode_number + 1) % config["acum_gradients"] == 0:
            policy_optimizer.step()
            policy_optimizer.zero_grad()
            critic_optimizer.step()
            critic_optimizer.zero_grad()
        if (episode_number + 1) % config["scheduler_step"] == 0:
            policy_scheduler.step()
            critic_scheduler.step()
        # Logging
        discounted_rewards = returns[0].item()
        metrics = {
            "Episode": episode_number,
            "Policy Avg Prob": policy_log_probs.exp().mean().item(),
            "Discounted rewards": discounted_rewards,
            "Rewards": sum(rewards),
            "Policy Loss": policy_loss.item(),
            "Critic Loss": critic_loss.item(),
            "Policy LR": policy_scheduler.get_last_lr()[0],
            "Critic LR": critic_scheduler.get_last_lr()[0],
        }
        # episode_progress_bar.set_postfix(metrics)
        run.log(metrics)
        if (episode_number + 1) % config["checkpoint_step"] == 0:
            save_checkpoint(
                str(checkpoint_dir / f"{run.name}-episode-{episode_number}.pt"),
                agent, policy_optimizer, critic_optimizer, episode_number + 1
            )
    save_checkpoint(
        str(checkpoint_dir / f"{run.name}-last.pt"),
        agent, policy_optimizer, critic_optimizer, episode_number + 1
    )
    env.close()
    run.finish()


def parse_cli_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="Configuration file (.yaml)")
    parser.add_argument(
        "-d",
        "--device",
        default="cpu",
        required=False,
        help="Pytorch device type",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints/", required=False, help="Dir to store runs"
    )
    parser.add_argument(
        '-i', '--id', required=False, default=None, help="Run name to use (if exists, it resumes the run)"
    )
    parser.add_argument(
        '-p', '--pretrained', required=False, default=None, help="Path to pretrained model (.pt)"
    )
    parser.add_argument(
        '--disable-wandb', action='store_true', help="Path to pretrained model (.pt)"
    )
    args = parser.parse_args()
    return args


def parse_config(filename: str):
    with open(filename, "r") as file:
        data = yaml.safe_load(file)
    return data


def collate_episode(
    episode: tuple[list, list, list, list, list],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs, actions, rewards, terminated, truncated = episode
    # All will have shape (N, T, ?)
    obs = torch.stack([torch.from_numpy(x).float() for x in obs])
    rewards = torch.tensor(rewards).reshape(-1, 1)
    actions = torch.tensor(actions).reshape(-1, 1)
    terminated = torch.tensor(terminated).reshape(-1, 1)
    truncated = torch.tensor(truncated).reshape(-1, 1)
    return obs, actions, rewards, terminated, truncated


def get_n_step_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    discount_factor: float,
    n: int,
) -> torch.Tensor:
    assert rewards.ndim == 2
    assert values.ndim == 2
    assert terminated.ndim == 2
    assert rewards.size(-1) == 1
    assert values.size(-1) == 1
    assert terminated.size(-1) == 1
    T = rewards.size(0)
    rewards = rewards.to(values.device)
    terminated = terminated.to(values.device)
    assert rewards.shape == terminated.shape == values[:-1].shape
    values = torch.concat([
        values[:1, :],
        torch.where(terminated, 0, values[1:, :])
    ])
    returns = torch.zeros_like(rewards)
    if n < 0:
        returns[T - 1] = rewards[T - 1] + discount_factor * values[T]
        for t in reversed(range(T - 1)):
            returns[t] = rewards[t] + discount_factor * returns[t + 1]
        return returns.detach()
    else:
        acum_discount_factor = torch.tensor(
            [discount_factor**t for t in range(n)], device=values.device
        ).reshape(-1, 1)
        for t in range(T):
            n_prime = min(n, T - t)
            discounted_rewards = (
                rewards[t : t + n_prime] * acum_discount_factor[:n_prime]
            ).sum()
            discounted_terminal_value = (values[t + n_prime] * discount_factor ** n_prime).item()
            returns[t, 0] = discounted_rewards + discounted_terminal_value
        return returns.detach()


def save_checkpoint(filename: str, agent, policy_optimizer, critic_optimizer, episodes: int):
    checkpoint = {
        "policy_state_dict": agent.policy.state_dict(),
        "policy_optimizer_state_dict": policy_optimizer.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "episodes": episodes,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filename: str, agent, policy_optimizer, critic_optimizer):
    checkpoint = torch.load(filename, weights_only=False)
    agent.policy.load_state_dict(checkpoint['policy_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
    critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    return checkpoint['episodes']

if __name__ == "__main__":
    main()
