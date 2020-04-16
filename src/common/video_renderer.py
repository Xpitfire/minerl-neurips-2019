import time
import os
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont
from lighter.decorator import context


class VideoRenderer(object):
    @context
    def __init__(self, fps=30):
        self.recording = []
        self.fps = fps
        self.cumulative_reward = 0

    def render(self):
        frames = []
        cumulative_reward = 0
        for i, (obs, inventory, action, reward) in enumerate(self.recording):
            canvas = Image.new(mode="RGB", size=(272, 288))
            canvas.paste(Image.fromarray(obs['pov']), box=(10, 10))
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default()
            white = (255, 255, 255)

            # used to exclude in TreeChop env
            if 'inventory' is not None:
                draw.text((110, 10), "coal: {}".format(inventory['coal']), white, font=font)
                draw.text((110, 20), "cobblestone: {}".format(inventory['cobblestone']), white, font=font)
                draw.text((110, 30), "crafting_table: {}".format(inventory['crafting_table']), white, font=font)
                draw.text((110, 40), "dirt: {}".format(inventory['dirt']), white, font=font)
                draw.text((110, 50), "furnace: {}".format(inventory['furnace']), white, font=font)
                draw.text((110, 60), "iron_axe: {}".format(inventory['iron_axe']), white, font=font)
                draw.text((110, 70), "iron_ingot: {}".format(inventory['iron_ingot']), white, font=font)
                draw.text((110, 80), "iron_ore: {}".format(inventory['iron_ore']), white, font=font)
                draw.text((110, 90), "iron_pickaxe: {}".format(inventory['iron_pickaxe']), white, font=font)
                draw.text((110, 100), "log: {}".format(inventory['log']), white, font=font)
                draw.text((110, 110), "planks: {}".format(inventory['planks']), white, font=font)
                draw.text((110, 120), "stick: {}".format(inventory['stick']), white, font=font)
                draw.text((110, 130), "stone: {}".format(inventory['stone']), white, font=font)
                draw.text((110, 140), "stone_axe: {}".format(inventory['stone_axe']), white, font=font)
                draw.text((110, 150), "stone_pickaxe: {}".format(inventory['stone_pickaxe']), white, font=font)
                draw.text((110, 160), "torch: {}".format(inventory['torch']), white, font=font)
                draw.text((110, 170), "wooden_axe: {}".format(inventory['wooden_axe']), white, font=font)
                draw.text((110, 180), "wooden_pickaxe: {}".format(inventory['wooden_pickaxe']), white, font=font)
                mainhand = obs['equipped_items']['mainhand']

                draw.text((110, 210), "damage: {}".format(mainhand['damage']), white, font=font)
                draw.text((110, 220), "maxDamage: {}".format(mainhand['maxDamage']), white, font=font)
                draw.text((110, 230), "type: {}".format(mainhand['type']), white, font=font)

            draw.text((10, 80), "frame: {}".format(i), white, font=font)

            draw.text((10, 100), "sprint: {}".format(action['sprint']), white, font=font)
            draw.text((10, 110), "attack: {}".format(action['attack']), white, font=font)
            draw.text((10, 120), "forward: {}".format(action['forward']), white, font=font)
            draw.text((10, 130), "back: {}".format(action['back']), white, font=font)
            draw.text((10, 140), "jump: {}".format(action['jump']), white, font=font)
            draw.text((10, 150), "left: {}".format(action['left']), white, font=font)
            draw.text((10, 160), "right: {}".format(action['right']), white, font=font)
            draw.text((10, 170), "sneak: {}".format(action['sneak']), white, font=font)
            draw.text((10, 190), "camera: {}".format(action['camera']), white, font=font)

            if 'place' in action:
                draw.text((10, 180), "place: {}".format(action['place']), white, font=font)
            if 'craft' in action:
                draw.text((10, 200), "craft: {}".format(action['craft']), white, font=font)
            if 'equip' in action:
                draw.text((10, 210), "equip: {}".format(action['equip']), white, font=font)
            if 'nearbyCraft' in action:
                draw.text((10, 220), "nearbyCraft: {}".format(action['nearbyCraft']), white, font=font)
            if 'nearbySmelt' in action:
                draw.text((10, 230), "nearbySmelt: {}".format(action['nearbySmelt']), white, font=font)

            cumulative_reward += reward
            draw.text((10, 250), "cumulative reward: {}".format(cumulative_reward), white, font=font)
            draw.text((10, 260), "reward: {}".format(reward), white, font=font)

            frames.append(canvas)
        return np.stack(frames, axis=0)

    def append(self, obs, inventory, action, reward):
        self.recording.append((obs, inventory, action, reward))

    def write(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        video = self.render()
        timestamp = time.strftime("%Y%m%d%H%M%S")
        video_file = "recording-{}.mp4".format(timestamp)
        video_path = os.path.join(path, video_file)
        imageio.mimwrite(video_path, video, fps=self.fps)
