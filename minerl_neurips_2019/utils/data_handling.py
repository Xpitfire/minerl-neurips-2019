# -*- coding: utf-8 -*-
"""General functions and classes for reading and accessing data
"""

import os
import json
from collections import OrderedDict
import numpy as np
import gym
import ffmpeg
from matplotlib import pyplot as plt


def get_frames(filename: str, start_frame: int, end_frame: int):
    """Extract subsequence of frames from a mp4 video as numpy array.
    Requires ffmpeg-python package (pip3 install ffmpeg-python).

    Parameters
    ----------
    filename : str
        Filepath to .mp4 file
    start_frame : int
        Start frame of subsequence
    end_frame : int
        End frame of subsequence

    Returns
    ---------
    frames : np.ndarray
        Extracted subsequence as numpy array of shape (end_frame-start_frame, 64, 64, 3) and datatype np.uint8
    """
    n_frames = end_frame - start_frame
    out, _ = (
        ffmpeg
        .input(filename)
        .filter('select', 'gte(n,{})'.format(start_frame))
        .output('pipe:', vframes=n_frames, format='rawvideo', pix_fmt='rgb24')
        .global_args('-loglevel', 'error')
        .run(capture_stdout=True)
    )
    frames = np.frombuffer(out, np.uint8).reshape(-1, 64, 64, 3)
    return frames


def get_actions(filename: str, start_action: int, end_action: int):
    """
    Extract subsequence of actions from the rendered.npz file.

    Parameters
    ----------
    filename : str
        Filepath to .npz file
    start_action : int
        Zero-based index of the first action in the subsequence to get
    end_action : int
        Zero-based index of the first action not in the subsequence to get
    """
    arrays = np.load(filename)
    return np.c_[tuple(arr[start_action:end_action]
                       for name, arr in arrays.items()
                       if name.startswith('action_'))]


def get_observations(filename: str, start_obs: int, end_obs: int):
    """
    Extract subsequence of observations from the rendered.npz file.

    Parameters
    ----------
    filename : str
        Filepath to .npz file
    start_obs : int
        Zero-based index of the first action in the subsequence to get
    end_obs : int
        Zero-based index of the first action not in the subsequence to get
    """
    arrays = np.load(filename)
    return np.c_[tuple(arr[start_obs:end_obs]
                       for name, arr in arrays.items()
                       if name.startswith('observation_'))]


def show_frames(frames: np.ndarray, title_prefix: str = '', pause: float = 0.25, speedup: bool = True):
    """Show frames in numpy array as video
    
    Parameters
    ----------
    frames: np.ndarray
        Numpy array of shape (end_frame-start_frame, 64, 64, 3) and datatype np.uint8
    title_prefix: str
        Prefix for plot axis title
    pause: float
        Pause between frames in seconds
    speedup: bool
        Reduce pause between all but last 10 frames to 0.001
    """
    im = plt.imshow(frames[0])
    plt.show()
    for f_i in range(frames.shape[0]):
        plt.title(f"{title_prefix}frame {f_i}/{frames.shape[0]}")
        im.set_data(frames[f_i])
        if speedup:
            if f_i > (frames.shape[0] - 10):
                plt.pause(pause)
            else:
                plt.pause(0.001)
        else:
            plt.pause(pause)


def get_metadata_score(dirname: str):
    """Return score (= -1 * duration_steps) for successful human demonstration sequence in dirname,
    return None if success==False
    
    Parameters
    ----------
    dirname: str
        Path to 'metadata.json' file
    
    Returns
    ----------
    score: int
        Demonstration score, which is (-1 * duration_steps) on success and None if not successful
    """
    meta_data_file_name = 'metadata.json'
    with open(os.path.join(dirname, meta_data_file_name), 'r') as f:
        metadata = json.load(f)
    if metadata['success']:
        return int(-metadata['duration_steps'])
    else:
        return None


class SpacesMapper(object):
    """Provides mapping functions for translating inventory-, equipment-, and action space to other spaces or item keys.
    See http://minerl.io/docs/environments/index.html#minerlobtaindiamond-v0 for more information on spaces.
    
    
    Parameters
    ----------
    envname: str
        Name of environment

    Methods
    ----------
    get_environment_spaces():
        Return lists of keys describing the inventory-, equipment-, and action space space keys, in that order.
    key_to_equipped(ind: int):
        Map item key to equipment space index
    key_to_inventory(ind: int):
        Map item key to inventory space index
    equipped_to_key(ind: int):
        Map equipment space equipment to item key
    inventory_to_key(ind: int):
        Map inventory space index to item key
    equipped_to_inventory(ind: int):
        Map equipment space index to inventory space index
    inventory_to_equipped(ind: int):
        Map inventory space index to equipment space index
    """
    
    def __init__(self, envname: str = "MineRLObtainDiamond-v0"):
        specs = gym.envs.registration.spec(envname)
        specs_kwargs = specs._kwargs
        equipped = specs_kwargs["observation_space"]["equipped_items"]["mainhand"]["type"].values
        inventory = [item for item in specs_kwargs["observation_space"]["inventory"].spaces.keys()]
        actions = OrderedDict()
        for action, values in specs_kwargs["action_space"].spaces.items():
            if "minerl.env.spaces.Discrete" in str(type(values)):
                actions[action] = {"type": "discrete", "values": list(range(values.n))}
            elif "minerl.env.spaces.Box" in str(type(values)):
                actions[action] = {"type": "box", "values": [values.low, values.high]}
            elif "minerl.env.spaces.Enum" in str(type(values)):
                actions[action] = {"type": "enum", "values": values.values}
        
        self.specs = specs
        self.equipped = equipped
        self.inventory = inventory
        self.actions = actions
    
    def get_environment_spaces(self):
        """Return lists of keys describing the inventory-, equipment-, and action space space keys, in that order.

        Returns
        -------
        inventory : list of str
            Keys in inventory space as list of str
        equipment : list of str
            Keys in equipment space as list of str
        action : list of str
            Keys in action space as list of str
        """
        return self.inventory, self.equipped, self.actions

    def key_to_equipped(self, key: str):
        """Map item key to equipment space index
        
        Parameters
        ----------
        key: int
            Item key

        Returns
        -------
        ind: int
            Equipment space index
        """
        try:
            return self.equipped.index(key)
        except ValueError:
            return None
        
    def key_to_placed(self, key: str):
        """Map item key to 'place' action space index
        
        Parameters
        ----------
        key: int
            Item key

        Returns
        -------
        ind: int
            'place' action space index
        """
        try:
            return self.actions["place"]["values"].index(key)
        except ValueError:
            return None

    def key_to_inventory(self, key: str):
        """Map item key to inventory space index
        
        Parameters
        ----------
        key: int
            Item key

        Returns
        -------
        ind: int
            Inventory space index
        """
        try:
            return self.inventory.index(key)
        except ValueError:
            return None
    
    def equipped_to_key(self, ind: int):
        """Map equipment space index to item key

        Parameters
        ----------
        ind: int
            Equipment space index

        Returns
        -------
        key: str
            Item key
        """
        return self.equipped[ind]
    
    def placed_to_key(self, ind: int):
        """Map 'place' action space index to item key

        Parameters
        ----------
        ind: int
            'place' action space index

        Returns
        -------
        key: str
            Item key
        """
        return self.actions["place"]["values"][ind]
    
    def inventory_to_key(self, ind: int):
        """Map inventory space index to item key

        Parameters
        ----------
        ind: int
            Inventory space index

        Returns
        -------
        key: str
            Item key
        """
        return self.inventory[ind]
    
    def equipped_to_inventory(self, ind: int):
        """Map equipment space index to inventory space index

        Parameters
        ----------
        ind: int
            Equipment space index

        Returns
        -------
        inventory: int
            Inventory space index
        """
        return self.key_to_inventory(self.equipped_to_key(ind))
    
    def inventory_to_equipped(self, ind: int):
        """Map inventory space index to equipment space index

        Parameters
        ----------
        ind: int
            Inventory space index

        Returns
        -------
        inventory: int
            Equipment space index
        """
        return self.key_to_equipped(self.inventory_to_key(ind))
    
    def placed_to_inventory(self, ind: int):
        """Map 'place' action space index to inventory space index

        Parameters
        ----------
        ind: int
            'place' action space index

        Returns
        -------
        inventory: int
            Inventory space index
        """
        return self.key_to_inventory(self.placed_to_key(ind))
    
    def inventory_to_placed(self, ind: int):
        """Map inventory space index to 'place' action space index

        Parameters
        ----------
        ind: int
            Inventory space index

        Returns
        -------
        inventory: int
            'place' action space index
        """
        return self.key_to_placed(self.inventory_to_key(ind))