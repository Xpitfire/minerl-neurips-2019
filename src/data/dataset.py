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
        self.batch_size = self.config.settings.bc.batch_size
        # add one additional sample for prediction in behavioural cloning
        self.seq_len = self.config.settings.model.seq_len
        self.slice_len = self.seq_len*(self.config.settings.env.frame_skip+1) + 1
        self.generator = self.data.sarsd_iter(num_epochs=-1, max_sequence_len=self.slice_len)
        self.consts = {
            "MineRLObtainDiamond-v0": 1836040.,
            "MineRLObtainDiamondDense-v0": 1836040.,
            "MineRLTreechop-v0": 439928.
        }
        self.iterations = self.consts[self.config.settings.env.env] / (self.slice_len*self.batch_size)
        self.itr = 0

    def sample(self):
        states = []
        inventory = []
        actions = []
        rewards = []
        for i in range(self.batch_size):
            state, action, reward, next_state, done = next(self.generator)
            # frame skip
            if self.config.settings.env.frame_skip > 0:
                raise NotImplementedError
            # skip too short sequences
            if len(done) < self.slice_len:
                continue

            i = self.transform.preprocess_inventory(state)
            inventory.append(i)

            o = self.transform.preprocess_states(state)
            o = self.transform.normalize_state(o)
            o = self.transform.reshape_states(o)
            states.append(o)

            a = self.transform.preprocess_camera(action)
            actions.append(a)
            rewards.append(reward)

        obs = torch.from_numpy(np.stack(states, axis=0))
        obs = obs.to(self.device)[:, :-1, ...]
        inventory = torch.from_numpy(np.stack(inventory, axis=0)).to(self.device)
        shape = (inventory.shape[0] * self.seq_len, ) + inventory.shape[2:]
        inventory = inventory.reshape(shape)
        actions = self.transform.stack_actions(actions, self.seq_len)
        rewards = torch.from_numpy(np.stack(rewards, axis=0)[..., np.newaxis]).to(self.device)[:, 1:, ...]
        shape = (rewards.shape[0] * self.seq_len, ) + rewards.shape[2:]
        rewards = rewards.reshape(shape)

        return {
            'obs': obs,
            'inventory': inventory,
            'actions': actions,
            'values': rewards
        }

    def __iter__(self):
        self.itr = 0
        return self

    def __next__(self):
        if self.itr >= self.iterations:
            raise StopIteration
        s = self.sample()
        self.itr += 1
        return s
