import numpy as np
import pathlib
import os
import minerl


def load_data(data_dir, env_name, force_download, **kwargs):
    if not os.path.exists(data_dir):
        pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(os.path.join(data_dir, 'VERSION')) or force_download:
        minerl.data.download(data_dir)
    return minerl.data.make(env_name, data_dir=data_dir, num_workers=1)


def data_wrapper(dataloader):
    def wrapper(epochs, batch_size, seed, sequence_len, add_noop=False):
        states = []
        actions = []
        cameras = []
        rewards = []

        batch_cnt = 0

        for obs, action, reward, next_state, done in \
                dataloader.sarsd_iter(num_epochs=epochs, seed=seed, max_sequence_len=sequence_len):

            # skip too short sequences
            if np.shape(obs['pov'])[0] != sequence_len:
                continue
            # swap RGB channel for conv layers
            s = transforms(obs['pov'])
            states.append(s)
            attack = action['attack'][..., np.newaxis]
            a = np.concatenate((attack,
                                action['forward'][..., np.newaxis],
                                action['back'][..., np.newaxis],
                                action['jump'][..., np.newaxis],
                                action['left'][..., np.newaxis],
                                action['right'][..., np.newaxis],
                                action['sneak'][..., np.newaxis],
                                action['sprint'][..., np.newaxis]), axis=-1)

            # if no action is selected, set noop
            if add_noop:
                noop = np.sum(a, axis=-1)
                empty = (noop == 0).astype(np.float32)[..., np.newaxis]
                a = np.concatenate((a, empty), axis=-1)

            # create batches
            actions.append(a)
            cameras.append(action['camera'])
            rewards.append(reward)
            batch_cnt += 1
            if batch_cnt == batch_size:
                # redistribute the reward to the last instance
                yield np.stack(states, axis=0).astype(np.float32), \
                      np.stack(actions, axis=0).astype(np.float32), \
                      np.stack(cameras, axis=0).astype(np.float32), \
                      np.stack(rewards, axis=0).astype(np.float32)[..., np.newaxis]
                batch_cnt = 0
                states = []
                actions = []
                cameras = []
                rewards = []
        # redistribute the reward to the last instance
        yield np.stack(states, axis=0).astype(np.float32), \
              np.stack(actions, axis=0).astype(np.float32), \
              np.stack(cameras, axis=0).astype(np.float32), \
              np.stack(rewards, axis=0).astype(np.float32)[..., np.newaxis]
    return wrapper


def value_transforms(rewards):
    new_rewards = []
    for r_seq in rewards:
        new_reward_seq = []
        for i, r in enumerate(r_seq):
            tmp = []
            for _ in range(i):
                tmp.append(0.0)
            for j in range(len(r_seq)-i):
                tmp.append(r * 0.9**(j+1))
            new_reward_seq.append(tmp)
        new_rewards.append(np.sum(np.array(new_reward_seq), axis=0) + np.array(r_seq))
    return new_rewards


def transforms(obs):
    x = obs.transpose(0, 3, 1, 2).astype(dtype=np.float)
    x /= 255.
    return x

