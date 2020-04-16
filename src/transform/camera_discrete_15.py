import torch
import numpy as np
from src.transform.base_transform import BaseTransform


class Transform(BaseTransform):
    def __init__(self):
        super(Transform, self).__init__()
        self.camera_mapping = {
            0: -18.0,
            1: -7.0,
            2: -5.0,
            3: -2.0,
            4: -1.0,
            5: -0.3,
            6: -0.1,
            7: 0.0,
            8: 0.1,
            9: 0.3,
            10: 1.0,
            11: 2.0,
            12: 5.0,
            13: 7.0,
            14: 18.0
        }
        self.bins = np.array([-np.inf, -18.0, -7.0, -5.0, -2.0, -1.0, -0.3, -0.1, 0.1, 0.3, 1.0, 2.0, 5.0, 7.0, 18.0, np.inf])

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
        camera = np.digitize(action['camera'], self.bins, right=True)
        action['camera1'] = camera[:, 0] - 1
        action['camera2'] = camera[:, 1] - 1
        return action
