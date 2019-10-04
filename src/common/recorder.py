import datetime
import pathlib
import time
import os
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
        self.values = []
        self.actions = []
        self.action_binaries = []

    def append_value(self, value: str):
        if self.recording:
            self.values.append(value)

    def append_action(self, action: str):
        if self.recording:
            self.actions.append(action)

    def append_action_binaries(self, action_binaries):
        if self.recording:
            self.action_binaries.append(action_binaries)

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
            st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S')
            if name_tag is not None:
                st += "-{}".format(name_tag)
            name = os.path.join(self.rec_save_dir, 'rec-{}'.format(st))
            video_frames = self.get_values_video(self.frames)
            if len(self.action_binaries) > 0:
                video_frames = self.get_binaries_video(video_frames)
            imageio.mimwrite('{}.mp4'.format(name), video_frames, fps=20)

    def get_binaries_video(self, frames):
        action_info = np.zeros((frames.shape[0], 16, 64, 3), dtype=np.uint8)
        for s in range(frames.shape[0]):
            for i_ba, ba in enumerate(range(len(self.action_binaries[0]))):
                idx = i_ba * 8
                if self.action_binaries[s][ba]:
                    action_info[s, :, idx:idx+8, :] = 255
            for i_ba in range(len(self.action_binaries)+1):
                idx = i_ba * 8
                action_info[s, :, idx:idx+1, :] = 128
        return np.concatenate((frames, action_info), axis=1)

    def get_values_video(self, frames):
        video = []
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame, 'RGB')
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            if len(self.values) > 0:
                draw.text((0, 0), "Val: {}".format(self.values[i]), (255, 255, 255), font=font)
            if len(self.actions) > 0:
                draw.text((0, 10), "Act: {}".format(self.actions[i]), (255, 255, 255), font=font)
            video.append(img)
        return np.stack(video, axis=0)
