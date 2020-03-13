import gym
gym.logger.set_level(40)
import time
import os
import imageio
import minerl
import numpy as np
from gym.wrappers import TimeLimit
from gym.wrappers.monitor import Monitor
from gym.wrappers.frame_stack import FrameStack
from lighter.decorator import context, reference
from src.envs.env_client import RemoteGym


class StateWrapper(gym.Wrapper):
    @context
    @reference(name='transform')
    def __init__(self, env):
        super(StateWrapper, self).__init__(env)
        self.input_dim = self.config.settings.model.input_dim
        self.input_channels = self.config.settings.model.input_channels
        shape = (self.input_dim, self.input_dim, self.input_channels)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        self.config.settings.env.observation_space = self.observation_space

    def step(self, action):
        obs, reward, done, info = super().step(action)
        obs = self.transform.preprocess_state(obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        obs = self.transform.preprocess_state(obs)
        return obs


class RecordingWrapper(gym.Wrapper):
    @context
    def __init__(self, env):
        super(RecordingWrapper, self).__init__(env)
        self.frames = []
        self.record_dir = 'video0'
        self.fps = self.config.settings.env.record_fps
        self.cumulative_reward = 0

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.frames.append(obs[..., 1:])
        self.cumulative_reward += reward
        if done:
            video = np.stack(self.frames, 0)
            timestamp = time.strftime("%Y%m%d%H%M%S")
            video_file = "recording-{}-reward-{}.mp4".format(timestamp, self.cumulative_reward)
            video_path = os.path.join(self.record_dir, video_file)
            imageio.mimwrite(video_path, video, fps=self.fps)
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.frames = []
        obs = super().reset(**kwargs)
        self.frames.append(obs[..., 1:])
        return obs


class NormalizationWrapper(gym.Wrapper):
    @context
    @reference(name='transform')
    def __init__(self, env):
        super(NormalizationWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_space.shape, dtype=np.float32)
        self.config.settings.env.observation_space = self.observation_space

    def step(self, action):
        obs, reward, done, info = super().step(action)
        obs = self.transform.normalize_state(obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        obs = self.transform.normalize_state(obs)
        return obs


class ReshapeWrapper(gym.Wrapper):
    @context
    @reference(name='transform')
    def __init__(self, env):
        super(ReshapeWrapper, self).__init__(env)
        low = 0
        high = 1
        dtype = np.float32
        shape = self.observation_space.shape
        shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
        self.config.settings.env.observation_space = self.observation_space

    def step(self, action):
        obs, reward, done, info = super().step(action)
        obs = self.transform.reshape_state(obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        obs = self.transform.reshape_state(obs)
        return obs


class VecEnv(object):
    def __init__(self, env_func, poolsize):
        assert poolsize > 0
        self.poolsize = poolsize
        self.envs = [env_func(i) for i in range(poolsize)]
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

    def __len__(self):
        return self.poolsize

    def render(self):
        return [e.render() for e in self.envs]

    def reset(self, **kwargs):
        return [e.reset(**kwargs) for e in self.envs]

    def step(self, action):
        obs = []
        reward = []
        done = []
        info = []
        for k, e in enumerate(self.envs):
            o, r, d, i = e.step(action[k])
            obs.append(o)
            reward.append(r)
            done.append(d)
            info.append(i)
        return obs, reward, done, info

    def close(self):
        [e.close() for e in self.envs]

    def values(self, value):
        return np.array([value for _ in range(self.poolsize)])

    def values_lambda(self, delegate):
        return [delegate() for _ in range(self.poolsize)]

    def noops(self):
        return [self.action_space.noop() for _ in range(self.poolsize)]


class EnvWrapper(object):
    @context
    def __init__(self):
        self.wrapper = RemoteGym(host=self.config.settings.env.host, port=self.config.settings.env.port)
        self.wrapper.connect()

    def _build(self, env_id):
        env = self.wrapper.make(self.config.settings.env.env)
        env = TimeLimit(env, self.config.settings.env.max_episode_steps)
        env = StateWrapper(env)
        if self.config.settings.env.record_video:
            env = RecordingWrapper(env)
            env = Monitor(env, directory='./video{}'.format(env_id), force=True)
        env = NormalizationWrapper(env)
        env = ReshapeWrapper(env)
        env = FrameStack(env, self.config.settings.model.seq_len)
        return env

    @property
    def env(self):
        env = VecEnv(self._build, self.config.settings.env.poolsize)
        return env
