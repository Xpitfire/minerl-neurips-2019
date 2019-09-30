import torch
import random
import numpy as np
import pathlib
import imageio
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


class VideoRecorder:
    def __init__(self, rec_save_dir=None, recording=False, **kwargs):
        if rec_save_dir is None:
            recording = False
            print('Recording disabled due to None save_dir')
        elif rec_save_dir is not None and recording:
            pathlib.Path(rec_save_dir).mkdir(parents=True, exist_ok=True)
            print('Recording enabled')
        else:
            print('Recording disabled')

        self.rec_save_dir = rec_save_dir
        self.recording = recording
        self.frames = []

    def append_frame(self, frame):
        if self.recording:
            # requires format size x size x color channel
            self.frames.append(frame)

    def append_frames(self, frames):
        if self.recording:
            for i in range(np.shape(frames)[0]):
                self.append_frame(frames[i, ...])

    def write(self, name_tag: str = None):
        if self.recording:
            print('Writing video recording...')
            frames = np.stack(self.frames, axis=0)
            st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S')
            if name_tag is not None:
                st += "-rew-{}".format(name_tag)
            imageio.mimwrite(os.path.join(self.rec_save_dir, "rec-{}.mp4".format(st)), frames, fps=20)


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
