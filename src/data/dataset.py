import minerl
import numpy as np
import torch
from lighter.decorator import context, reference, device


class DataGenerator(object):
    @device
    @context
    @reference(name='transform')
    def __init__(self):
        self.data = minerl.data.make(self.config.settings.env.env,
                                     data_dir=self.config.settings.env.data_root,
                                     force_download=True)
        self.generator = None
        self.batch_size = self.config.settings.bc.batch_size
        # add one additional sample for prediction in behavioural cloning
        self.seq_len = self.config.settings.model.seq_len + 1

    def sample(self):
        states = []
        actions = []
        rewards = []
        for i in range(self.batch_size):
            state, action, reward, next_state, done = next(self.generator)
            if len(done) < self.seq_len:
                continue
            o = self.transform.preprocess_states(state)
            o = self.transform.normalize_state(o)
            o = self.transform.reshape_states(o)
            states.append(o)

            a = self.transform.discretize_camera(action)
            actions.append(a)
            rewards.append(reward)

        obs = torch.from_numpy(np.stack(states, axis=0)).to(self.device)[:, :-1, ...]
        actions = self.transform.stack_actions(actions)
        rewards = torch.sum(torch.from_numpy(np.stack(rewards, axis=0)[..., np.newaxis]).to(self.device), dim=1)
        return {
            'obs': obs,
            'actions': actions,
            'values': rewards
        }

    def init(self):
        self.generator = self.data.sarsd_iter(num_epochs=-1, max_sequence_len=self.seq_len)
