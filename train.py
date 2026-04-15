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
from environment import Layout
from agent import RandomAgent, DRLAgent


def main():
    cli_args = parse_cli_args()
    config = parse_config(cli_args.config)
    device = cli_args.device
    load_dotenv()
    run = wandb.init(
        entity="sarambulo-carnegie-mellon-university",
        project="cogrob",
        config=config,
    )

    # Environment setup
    env = gym.make(
        "GridEnv",
        max_episode_steps=config["max_episode_steps"],
        layout=Layout(),
        render_mode="rgb_array",
    )
    record_episode = lambda x: (x + 1) % config["record_every"] == 0
    env = gym.wrappers.RecordVideo(env, "runs", episode_trigger=record_episode, name_prefix="train")
    # Agent setup
    agent = DRLAgent(action_space=env.action_space, device=device)
    # Learning setup
    policy_optimizer = torch.optim.AdamW(agent.policy.parameters(), lr=config["policy_lr"])
    critic_optimizer = torch.optim.AdamW(agent.critic.parameters(), lr=config["critic_lr"])
    policy_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        policy_optimizer, gamma=config["policy_lr_decay"]
    )
    critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        critic_optimizer, gamma=config["critic_lr_decay"]
    )
    episode_progress_bar = tqdm(range(config["train_episodes"]), desc="Training episodes")
    for episode in episode_progress_bar:
        observations, actions, rewards, terminated, truncated = agent.run(env)
        returns = get_returns(rewards, discount_factor=config["discount_factor"]).to(device)
        logits = agent.action_logits(observations[:-1])
        actions_tensor = torch.tensor(actions).to(device).unsqueeze(-1)
        assert len(logits) == len(actions_tensor)
        policy_log_probs = torch.log_softmax(logits, dim=-1).gather(dim=-1, index=actions_tensor)
        values = agent.values(observations)
        terminated_tensor = torch.tensor(terminated).to(device)
        rewards_tensor = torch.tensor(rewards).float().to(device)
        advantages = get_advantages(
            values, rewards_tensor, terminated_tensor, discount_factor=config["discount_factor"]
        )
        policy_loss = (-advantages.detach() * policy_log_probs).mean()
        critic_loss = torch.nn.functional.mse_loss(values[:-1], returns.detach())
        policy_loss.backward()
        critic_loss.backward()
        if (episode + 1) % config["acum_gradients"] == 0:
            policy_optimizer.step()
            policy_optimizer.zero_grad()
            critic_optimizer.step()
            critic_optimizer.zero_grad()
        if (episode + 1) % config["scheduler_step"] == 0:
            policy_scheduler.step()
            critic_scheduler.step()
        # Logging
        discounted_rewards = returns[0].item()
        metrics = {
            "Episode": episode,
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
    run.finish()
    checkpoint_dir = Path(cli_args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    save_checkpoint(str(checkpoint_dir / "last.pt"), agent, policy_optimizer, critic_optimizer)


def parse_cli_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="Configuration file (.yaml)")
    parser.add_argument("-d", "--device", default="cpu", required=False, help="Pytorch device type", choices=['cpu', 'cuda'])
    parser.add_argument("--checkpoint-dir", default="checkpoints/", required=False, help="Dir to store runs")
    args = parser.parse_args()
    return args


def parse_config(filename: str):
    with open(filename, "r") as file:
        data = yaml.safe_load(file)
    return data


def get_advantages(
    values: torch.Tensor, rewards: torch.Tensor, terminated: torch.Tensor, discount_factor: float
) -> torch.Tensor:
    # A_t = Q_t - V_t
    # Q_t = r_t + V_t+1
    values_t = values[:-1]
    if len(values) > 1:
        values_next_t = values[1:]
    else:
        values_next_t = torch.zeros_like(rewards)
    assert rewards.shape == terminated.shape == values_t.shape == values_next_t.shape
    values_next_t = torch.where(terminated, 0, values_next_t)
    advantages = rewards + values_next_t * discount_factor - values_t
    return advantages.detach()


def get_returns(rewards: np.ndarray | list, discount_factor: float) -> torch.Tensor:
    T = len(rewards)
    returns = torch.zeros((T,)).float()
    returns[T - 1] = rewards[T - 1]
    for t in range(T - 2, -1, -1):
        returns[t] = rewards[t] + returns[t + 1] * discount_factor
    return returns.detach()


def save_checkpoint(filename: str, agent, policy_optimizer, critic_optimizer):
    checkpoint = {
        "policy_state_dict": agent.policy.state_dict(),
        "policy_optimizer_state_dict": policy_optimizer.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


if __name__ == "__main__":
    main()
