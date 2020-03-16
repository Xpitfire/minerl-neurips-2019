import minerl
import numpy as np
import torch
from lighter.decorator import context, reference, device


class DataGenerator(object):
    @device
    @context
    @reference(name='transform')
    def __init__(self, epochs=1):
        self.data = minerl.data.make(self.config.settings.env.env,
                                     data_dir=self.config.settings.env.data_root,
                                     force_download=True)
        self.batch_size = self.config.settings.bc.batch_size
        # add one additional sample for prediction in behavioural cloning
        self.seq_len = self.config.settings.model.seq_len*(self.config.settings.env.frame_skip+1) + 1
        self.epochs = epochs
        self.generator = self.data.sarsd_iter(num_epochs=self.epochs, max_sequence_len=self.seq_len)
        self.consts = {
            "MineRLObtainDiamond-v0": 1836040,
            "MineRLObtainDiamondDense-v0": 1836040,
            "MineRLTreechop-v0": 439928
        }

    def sample(self):
        states = []
        actions = []
        rewards = []
        for i in range(self.batch_size):
            state, action, reward, next_state, done = next(self.generator)
            # frame skip
            if self.config.settings.env.frame_skip > 0:
                raise NotImplementedError
            # skip too short sequences
            if len(done) < self.seq_len:
                continue

            o = self.transform.preprocess_states(state)
            o = self.transform.normalize_state(o)
            o = self.transform.reshape_states(o)
            states.append(o)

            a = self.transform.discretize_camera(action)
            actions.append(a)
            rewards.append(reward)

        obs = torch.from_numpy(np.stack(states, axis=0))
        obs = obs.to(self.device)[:, :-1, ...]
        actions = self.transform.stack_actions(actions)
        rewards = torch.sum(torch.from_numpy(np.stack(rewards, axis=0)[..., np.newaxis]).to(self.device), dim=1)

        return {
            'obs': obs,
            'actions': actions,
            'values': rewards
        }

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample()
