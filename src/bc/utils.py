import torch
import random
import numpy as np
import datetime
import time
import os


def get_device(gpu, **kwargs):
    gpu = "cuda:" + gpu
    return torch.device(gpu if torch.cuda.is_available() else 'cpu')


def set_seeds(seed=None, only_env_seed=None):
    if seed is not None:
        if only_env_seed is not None and only_env_seed:
            return
        print('Set global seeds')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(model, checkpoint=None, device=None, **kwargs):
    if checkpoint is not None:
        print("load model from checkpoint...")
        model.load_state_dict(torch.load(checkpoint, map_location=device))


def save_model(model, save_dir, step):
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S')
    torch.save(model.state_dict(), os.path.join(save_dir, "checkpoint-net-{}-{}.tar".format(st, step)))
    torch.save(model.cnn.state_dict(), os.path.join(save_dir, "checkpoint-cnn-{}-{}.tar".format(st, step)))
    torch.save(model.cnn_lstm.state_dict(), os.path.join(save_dir, "checkpoint-cnn_lstm-{}-{}.tar".format(st, step)))
