import gym
import minerl
import os
import monitor
import global_params

# Set minerl download directory for expert data
os.environ['MINERL_DATA_ROOT'] = "/home/xpitfire/workspace/data/minerl"
# Set global parameters and logging (tensorboard)
args = global_params.args
args.enable_logging = False
logger = monitor.Logger(args)
# Define current environment name
env_name = "MineRLTreechop-v0"


def create_env(env_n):
    # Create a new environment
    return gym.make(env_n)


def play_episode(env):
    # Reset the environment
    obs, _ = env.reset()
    done = False
    while not done:
        # Take a no-op through the environment.
        obs, rew, done, _ = env.step(env.action_space.noop())
        # Do something


def load_expert_episode(env_n):
    # Load downloaded expert data
    return minerl.data.make(env_n)


def replay_expert_episode(data):
    # Iterate through a single epoch gathering sequences of at most 32 steps
    for obs, rew, done, act in data.seq_iter(num_epochs=1, max_sequence_len=32):
        print(act)
        # Do something


episode_data = load_expert_episode(env_name)
replay_expert_episode(episode_data)
