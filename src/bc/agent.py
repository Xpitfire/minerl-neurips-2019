import torch
import gym
import os
import numpy as np
from src.common.recorder import VideoRecorder
from src.common.data import transforms
from torch.distributions import Bernoulli


def build_agent(render_env=None, render=None, seed=None, **kwargs):
    if render is not None and render:
        print('Building env...')
        env_name = os.getenv('MINERL_GYM_ENV', render_env)
        env = gym.make(env_name)
        if seed is not None:
            print('Set environment seed')
            env.seed(seed)
            env.action_space.seed(seed)
        return Agent(env, **kwargs.copy())
    return None


class Agent:
    def __init__(self, env, sequence_len=None, repeated_env_reset=None, device=None):
        self.env = env
        self.sequence_len = sequence_len
        self.device = device
        self.repeated_reset = repeated_env_reset

        self.done = None
        self.model = None
        self.state_history = None
        self.noop = None
        self.reward = None
        self.value = None
        self.single_reset = None
        self.obs = None

    def reset(self):
        self.obs = self.env.reset()
        self.noop = self.env.action_space.noop
        state = transforms(self.obs['pov'][np.newaxis, ...])
        self.state_history = [state] * self.sequence_len
        self.reward = 0.0
        self.value = 0.0
        self.done = False

    def _get_state_history_tensor(self, state):
        self.state_history = self.state_history[1:] + [state]
        arr = np.stack(self.state_history, axis=1)
        return torch.from_numpy(arr).float().to(self.device)

    def _build_action(self, preds):
        action_pred, camera_pred, value_pred = preds

        # set action predictions
        action_probs = torch.sigmoid(action_pred)
        actions_dist = Bernoulli(probs=action_probs)
        actions = actions_dist.sample().squeeze().detach().cpu().numpy()
        template = self.noop()
        template['attack'] = int(actions[0])
        template['forward'] = int(actions[1])
        template['back'] = int(actions[2])
        template['jump'] = int(actions[3])
        template['left'] = int(actions[4])
        template['right'] = int(actions[5])
        template['sneak'] = int(actions[6])
        template['sprint'] = int(actions[7])

        # set camera predictions
        cam = camera_pred.squeeze().detach().cpu().numpy()
        template['camera'] = [cam[0], cam[1]]
        return template, (action_pred, camera_pred, value_pred, actions)

    def act(self, state):
        inputs = self._get_state_history_tensor(state)
        preds = self.model(inputs)
        return self._build_action(preds)

    def run(self, model, max_agent_steps=None, **kwargs):
        self.model = model
        recorder = VideoRecorder(**kwargs.copy())

        steps = 0
        if self.obs is None or self.repeated_reset or self.done:
            self.reset()

        while not self.done:
            state = transforms(self.obs['pov'][np.newaxis, ...])
            action, preds = self.act(state)
            _, _, values, action_binaries = preds
            obs, reward, done, info = self.env.step(action)
            self.obs = obs
            self.done = done
            self.reward += reward
            self.value = np.mean(values.detach().cpu().numpy())
            self.env.render()

            recorder.append_action_binaries(action_binaries)
            recorder.append_value(str(self.value))
            recorder.append_frame(self.obs['pov'])

            steps += 1
            if max_agent_steps is not None and steps >= max_agent_steps:
                self.done = True

        recorder.write('rew-{}_value-{}'.format(self.reward, self.value))
