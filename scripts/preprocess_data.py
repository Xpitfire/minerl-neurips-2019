import os
import numpy as np
from lighter.context import Context
from lighter.decorator import config
import skvideo.io
from src.common.video_renderer import VideoRenderer


class Preprocessor(object):
    @config(path='configs/preprocess_data.json')
    def __init__(self):
        self.root_dir = os.path.join(self.config.data_root, self.config.env)
        self.target_dir = self.config.target_root

    def build_inventory(self, pos, rendered):
        return {
            "coal": rendered['observation_inventory'][pos][0],
            "cobblestone": rendered['observation_inventory'][pos][1],
            "crafting_table": rendered['observation_inventory'][pos][2],
            "dirt": rendered['observation_inventory'][pos][3],
            "furnace": rendered['observation_inventory'][pos][4],
            "iron_axe": rendered['observation_inventory'][pos][5],
            "iron_ingot": rendered['observation_inventory'][pos][6],
            "iron_ore": rendered['observation_inventory'][pos][7],
            "iron_pickaxe": rendered['observation_inventory'][pos][8],
            "log": rendered['observation_inventory'][pos][9],
            "planks": rendered['observation_inventory'][pos][10],
            "stick": rendered['observation_inventory'][pos][11],
            "stone": rendered['observation_inventory'][pos][12],
            "stone_axe": rendered['observation_inventory'][pos][13],
            "stone_pickaxe": rendered['observation_inventory'][pos][14],
            "torch": rendered['observation_inventory'][pos][15],
            "wooden_axe": rendered['observation_inventory'][pos][16],
            "wooden_pickaxe": rendered['observation_inventory'][pos][17]
        }

    def build_obs(self, pos, video, rendered, inventory):
        return {
            "pov": video,
            "inventory": inventory,
            "equipped_items": {
                "mainhand": {
                    "damage": rendered['observation_damage'][pos],
                    "maxDamage": rendered['observation_maxDamage'][pos],
                    "type": rendered['observation_type'][pos]
                }
            }
        }

    def build_action(self, pos, rendered):
        return {
            "attack": rendered['action_attack'][pos],
            "back": rendered['action_back'][pos],
            "camera": rendered['action_camera'][pos],
            "craft": rendered['action_craft'][pos],
            "equip": rendered['action_equip'][pos],
            "forward": rendered['action_forward'][pos],
            "jump": rendered['action_jump'][pos],
            "left": rendered['action_left'][pos],
            "nearbyCraft": rendered['action_nearbyCraft'][pos],
            "nearbySmelt": rendered['action_nearbySmelt'][pos],
            "place": rendered['action_place'][pos],
            "right": rendered['action_right'][pos],
            "sneak": rendered['action_sneak'][pos],
            "sprint": rendered['action_sprint'][pos]
        }

    def read(self):
        offset_video = 130
        for path in os.listdir(self.root_dir):
            sub_dir = os.path.join(self.root_dir, path)
            rendered_file = os.path.join(sub_dir, 'rendered.npz')
            video_file = os.path.join(sub_dir, 'recording.mp4')
            video = self.read_video(video_file)
            video_renderer = VideoRenderer()
            with np.load(rendered_file, mmap_mode='r') as rendered:
                print(rendered.files)
                print('observation_damage', rendered['observation_damage'].shape)
                print('observation_maxDamage', rendered['observation_maxDamage'].shape)
                print('observation_type', rendered['observation_type'].shape)
                print('observation_inventory', rendered['observation_inventory'].shape)
                print('reward', rendered['reward'].shape)
                print('action_attack', rendered['action_attack'].shape)
                print('action_craft', rendered['action_craft'].shape)
                print('video', video.shape)
                for i in range(rendered['reward'].shape[0])[-1000:]:
                    inventory = self.build_inventory(i, rendered)
                    obs = self.build_obs(i, video[i + offset_video], rendered, inventory)
                    action = self.build_action(i, rendered)
                    reward = rendered['reward'][i]
                    video_renderer.append(obs, inventory, action, reward)
            video_renderer.render()
            video_renderer.write(self.target_dir)
            break

    def read_video(self, file):
        return skvideo.io.vread(file)


if __name__ == '__main__':
    Context.create()
    p = Preprocessor()
    p.read()
