import torch
import numpy as np
from src.transform.base_transform import BaseTransform


class Transform(BaseTransform):
    def __init__(self):
        super(Transform, self).__init__()

    def stack_actions(self, actions, seq_len):
        result = dict.fromkeys(actions[0].keys(), [])
        for k in result.keys():
            result[k] = []
            for a in actions:
                result[k].append(a[k])
        for k in result.keys():
            result[k] = torch.from_numpy(np.stack(result[k], axis=0)).to(self.device)[:, 1:, ...]
            shape = (result[k].shape[0] * seq_len, ) + result[k].shape[2:]
            result[k] = result[k].reshape(shape)

        result['move'] = torch.stack([
            result['attack'],
            result['forward'],
            result['jump'],
            result['back'],
            result['left'],
            result['right'],
            result['sneak'],
            result['sprint']
        ], dim=-1).float()

        return result

    def preprocess_camera(self, action):
        action['camera1'] = action['camera'][:, 0]
        action['camera2'] = action['camera'][:, 1]
        return action
