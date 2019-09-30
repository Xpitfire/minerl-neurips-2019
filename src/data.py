import numpy as np
import pathlib
import os
import minerl


means = np.array([0.485, 0.456, 0.406])[..., np.newaxis, np.newaxis]
std = np.array([0.229, 0.224, 0.225])[..., np.newaxis, np.newaxis]


def load_data(data_dir, env_name, force_download, **kwargs):
    if not os.path.exists(data_dir):
        pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(os.path.join(data_dir, 'VERSION')) or force_download:
        minerl.data.download(data_dir)
    return minerl.data.make(env_name, data_dir=data_dir, num_workers=1)


def data_wrapper(dataloader, trans):
    def wrapper(epochs, batch_size, seed, sequence_len, add_noop=False):
        states = []
        actions = []
        cameras = []

        batch_cnt = 0

        for obs, action, reward, next_state, done in \
                dataloader.sarsd_iter(num_epochs=epochs, seed=seed, max_sequence_len=sequence_len):

            # skip too short sequences
            if np.shape(obs['pov'])[0] != sequence_len:
                continue
            # swap RGB channel for conv layers
            s = trans(obs['pov'])
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
            batch_cnt += 1
            if batch_cnt == batch_size:
                yield np.stack(states, axis=0).astype(np.float32), \
                      np.stack(actions, axis=0).astype(np.float32), \
                      np.stack(cameras, axis=0).astype(np.float32)
                batch_cnt = 0
                states = []
                actions = []
                cameras = []

        yield np.stack(states, axis=0).astype(np.float32), \
              np.stack(actions, axis=0).astype(np.float32), \
              np.stack(cameras, axis=0).astype(np.float32)
    return wrapper


def transforms(obs):
    x = obs.transpose(0, 3, 1, 2)
    # use ImageNet statistics to normalize image
    x = x / 255.
    for i in range(np.shape(x)[0]):
        x[i] = x[i] - means
        x[i] = x[i] / std
    return x
