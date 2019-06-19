# -*- coding: utf-8 -*-
"""Subtask identification
"""

import os
import glob
import json
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from minerl_neurips_2019.utils.data_handling import get_metadata_score, get_frames, show_frames, SpacesMapper


def load_subtask_lookup(filename: str):
    """Load subtask lookup file to dictionary
    
    Parameters
    ----------
    filename: str
        Path to json subtask file, as created by make_lookup_dict_from_raw()
    
    Returns
    ----------
    lookup: dict
        Lookup dictionary with subtasks (item keys) as keys and list of associated demonstrations as values.
        Demonstrations are represented as tuple of (filename, score, start-frame, end-frame).
    """
    with open(filename, 'r') as f:
        lookup = json.load(f)
    return lookup


def show_subtask_sequences(json_path: str, task_id: str, replacings: tuple = (('rendered.npz', 'recording.mp4'),),
                           pause: float = 0.25):
    """Show frames of all subtasks in file json_path with task_id as video.
    
    Parameters
    ----------
    json_path: str
        Path to json subtask file, as created by make_lookup_dict_from_raw()
    task_id: str
        ID of subtask, e..g 'log', as created by make_lookup_dict_from_raw()
    replacings: tuple of tuple
        Tuple of string-pairs, where for each string pair the first string will be replaced by the second string in the
        filenpaths in json_path.
    pause: float
        Pause between last 10 frames in seconds.
    
    """
    with open(json_path, 'r') as f:
        lookup = json.load(f)
    lookup = lookup[task_id]
    
    for s_i, subseq in enumerate(lookup):
        fname = subseq[0]
        for replacing in replacings:
            fname = fname.replace(*replacing)
        frames = get_frames(fname, subseq[2], subseq[3]).copy()
        max_val = frames.max(axis=-1).max(axis=-1).max(axis=-1)
        max_val = max_val[:, None, None, None]
        frames = np.asarray((frames / max_val) * 255, dtype=np.uint8)
        show_frames(frames=frames, title_prefix=f"Sample {s_i}, ", pause=pause)
    

def make_lookup_dict_from_raw(input_dir: str, best_replay_frac: float = 1., make_files: bool = False):
    """Parse files in input_dir and creates:
    
      - inventory_changes, a dictionary with filenames as keys and tuples of numpy arrays (frames, item_ids) for each
        positive inventory changes
      - subtask_lookup, a list with elements corresponding to subtasks.
        Each element is a list of tuples (filename, score, start-frame, end-frame),
        one tuple for each demonstration sequence of the subtask.
    
    Parameters
    ----------
    input_dir : str
        Path to folder containing human demonstration files ('univ.json')
    best_replay_frac : float
        Only consider a fraction best_replay_frac of successful replays.
    make_files : bool
        If True, writes subtask_lookup to file 'subtask_lookup.json' and inventory_changes to file
        'inventory_changes.npz' in directory input_dir.
    
    Returns
    ----------
    inventory_changes : (OrderedDict of tuple, list of list)
      inventory_changes: a dictionary with filenames as keys and tuples of numpy arrays (frames, item_ids) for each
      positive inventory changes.
      subtask_lookup: a list with elements corresponding to subtasks.
      Each element is a list of tuples (filename, score, start-frame, end-frame),
      one tuple for each demonstration sequence of the subtask.
    """
    feature_file_name = 'rendered.npz'
    spaces_mapper = SpacesMapper()
    inv_space, *_ = spaces_mapper.get_environment_spaces()
    
    feature_files = glob.glob(os.path.join(input_dir, '**', feature_file_name))
    feature_files.sort()
    
    # Create list (filename, score) and restrict to successful replays only
    feature_files_and_scores = [(ff, get_metadata_score(os.path.dirname(ff))) for ff in feature_files
                                if get_metadata_score(os.path.dirname(ff)) is not None]
    
    # Sort by score, take only best_replay_frac
    feature_files_and_scores.sort(key=lambda x: x[1], reverse=True)
    feature_files_and_scores = feature_files_and_scores[:int(np.ceil(len(feature_files_and_scores)*best_replay_frac))]
    
    inventory_changes = OrderedDict()
    replay_scores = OrderedDict()
    subtask_lookup = OrderedDict([(item, []) for item in inv_space])
    
    for ff_path, score in tqdm(feature_files_and_scores):
        data = np.load(ff_path, allow_pickle=True, mmap_mode='r')
        inv = data['observation_inventory'].copy()
        equip = data['action_equip']
        place = data['action_place']
        # translate equip index to inv index
        for inv_i in range(inv.shape[-1]):
            equ_i = spaces_mapper.inventory_to_equipped(inv_i)
            if equ_i is not None:
                inv[:, inv_i] += equip == equ_i
        
        inv_changes = np.where((inv[1:] - inv[:-1]) > 0)
        # discard changes for previously placed items
        del_i = set()
        
        for plc_i in range(len(spaces_mapper.actions["place"]["values"])):
            if spaces_mapper.placed_to_inventory(plc_i) is not None:
                for plc_act in np.where(place == plc_i)[0]:    
                    x = inv_changes[0][inv_changes[1] == spaces_mapper.placed_to_inventory(plc_i)]
                    chng_after_plc = np.where(x > plc_act)[0]
                    if len(chng_after_plc) > 0:
                        x_i = chng_after_plc[0]            
                        del_i.add(np.where(inv_changes == x[x_i])[1][0])
        del_i = list(del_i)
        inv_changes = (np.delete(inv_changes[0], del_i), np.delete(inv_changes[1], del_i))
        # inc frame idx
        inv_changes[0][:] += 1
        
        # tuple of numpy arrays (frames, item_ids) per file for positive inventory changes
        inventory_changes[ff_path] = inv_changes
        replay_scores[ff_path] = score
        
        start_frame = 0
        for element_i, item_id in enumerate(inv_changes[1]):
            end_frame = inv_changes[0][element_i]
            subtask_lookup[spaces_mapper.inventory_to_key(item_id)].append((ff_path, score, int(start_frame),
                                                                            int(end_frame)))
            
            # If two frames are not equal, make end frame to new start frame, otherwise keep original start frame
            # (-> if multiple items were crafted in the same frame, items are treated as if there were the only
            #  items crafted at this frame)
            if ((element_i + 1) < len(inv_changes[0])) and (inv_changes[0][element_i] != inv_changes[0][element_i + 1]):
                start_frame = end_frame
    
    # Save results json files
    if make_files:
        np.savez(os.path.join(input_dir, 'inventory_changes.npz'), **inventory_changes)
        np.savez(os.path.join(input_dir, 'replay_scores.npz'), **replay_scores)
        with open(os.path.join(input_dir, 'subtask_lookup.json'), 'w') as o_f:
            json.dump(subtask_lookup, fp=o_f)
    
    return inventory_changes, subtask_lookup


if __name__ == '__main__':
    #
    # Parse files in input_dir and create
    # - inventory_changes, a dictionary with filenames as keys and tuples of numpy arrays (frames, item_ids) for each
    #   positive inventory changes
    # - subtask_lookup, a list with elements corresponding to subtasks.
    #   Each element is a list of tuples (filename, start-frame, end-frame),
    #   one tuple for each demonstration sequence of the subtask.
    #
    data_dir = '/publicdata/minerl/dataset_20190608/MineRLObtainDiamond-v0'
    
    changes, subtasks = make_lookup_dict_from_raw(input_dir=data_dir, make_files=True)
    
    loaded_subtasks = load_subtask_lookup(os.path.join(data_dir, 'subtask_lookup.json'))
    
    print("Done!")
