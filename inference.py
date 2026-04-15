import environment
import gymnasium as gym
from environment import Layout
from agent import RandomAgent, DRLAgent 

def main():
    env = gym.make("GridEnv", max_episode_steps=100, layout=Layout(), render_mode="rgb_array")
    record_episode = lambda x: True
    env = gym.wrappers.RecordVideo(
        env, "runs", episode_trigger=record_episode, name_prefix="inference"
    )
    agent = DRLAgent(action_space=env.action_space)
    episode = agent.run(env)
    print(sum(episode[2]))

if __name__=="__main__":
    main()