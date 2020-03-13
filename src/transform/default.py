import torch
import numpy as np
from lighter.decorator import context, device
from PIL import Image, ImageDraw, ImageFont


class Transform(object):
    @context
    @device
    def __init__(self):
        self.input_dim = self.config.settings.model.input_dim
        self.craft_mapping = {
            0: 0,
            1: "torch",
            2: "stick",
            3: "planks",
            4: "crafting_table"
        }
        self.equip_mapping = {
            0: 0,
            1: "air",
            2: "wooden_axe",
            3: "wooden_pickaxe",
            4: "stone_axe",
            5: "stone_pickaxe",
            6: "iron_axe",
            7: "iron_pickaxe"
        }
        self.nearbyCraft_mapping = {
            0: 0,
            1: "wooden_axe",
            2: "wooden_pickaxe",
            3: "stone_axe",
            4: "stone_pickaxe",
            5: "iron_axe",
            6: "iron_pickaxe",
            7: "furnace"
        }
        self.nearbySmelt_mapping = {
            0: 0,
            1: "iron_ingot",
            2: "coal"
        }
        self.place_mapping = {
            0: 0,
            1: "dirt",
            2: "stone",
            3: "cobblestone",
            4: "crafting_table",
            5: "furnace",
            6: "torch"
        }

    def prepare_for_model(self, x):
        return torch.from_numpy(np.stack(x, axis=0)).to(self.device)

    def prepare_for_env(self, output_dict, noops):
        for i in range(self.config.settings.env.poolsize):
            camera = output_dict['camera'][i].detach().cpu().numpy().squeeze()
            noops[i]['camera'] = [camera[0], camera[1]]
            noops[i]['attack'] = int(output_dict['move'][i][0].item())
            noops[i]['forward'] = int(output_dict['move'][i][1].item())
            noops[i]['back'] = int(output_dict['move'][i][2].item())
            noops[i]['jump'] = int(output_dict['move'][i][3].item())
            noops[i]['left'] = int(output_dict['move'][i][4].item())
            noops[i]['right'] = int(output_dict['move'][i][5].item())
            noops[i]['sneak'] = int(output_dict['move'][i][6].item())
            noops[i]['sprint'] = int(output_dict['move'][i][7].item())
            noops[i]['forward'] = int(output_dict['move'][i][1].item())
            noops[i]['craft'] = self.craft_mapping[output_dict['craft'][i].item()]
            noops[i]['equip'] = self.equip_mapping[output_dict['equip'][i].item()]
            noops[i]['place'] = self.place_mapping[output_dict['place'][i].item()]
            noops[i]['nearbyCraft'] = self.nearbyCraft_mapping[output_dict['nearbyCraft'][i].item()]
            noops[i]['nearbySmelt'] = self.nearbySmelt_mapping[output_dict['nearbySmelt'][i].item()]
        return noops

    def preprocess_state(self, obs):
        canvas = Image.new(mode="L", size=(self.input_dim, self.input_dim))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        white = (255,)

        inventory = obs['inventory']
        draw.text((0, 0), "{}".format(inventory['coal']), white, font=font)
        draw.text((16, 0), "{}".format(inventory['cobblestone']), white, font=font)
        draw.text((32, 0), "{}".format(inventory['crafting_table']), white, font=font)
        draw.text((48, 0), "{}".format(inventory['dirt']), white, font=font)
        draw.text((0, 10), "{}".format(inventory['furnace']), white, font=font)
        draw.text((16, 10), "{}".format(inventory['iron_axe']), white, font=font)
        draw.text((32, 10), "{}".format(inventory['iron_ingot']), white, font=font)
        draw.text((48, 10), "{}".format(inventory['iron_ore']), white, font=font)
        draw.text((0, 20), "{}".format(inventory['iron_pickaxe']), white, font=font)
        draw.text((16, 20), "{}".format(inventory['log']), white, font=font)
        draw.text((32, 20), "{}".format(inventory['planks']), white, font=font)
        draw.text((48, 20), "{}".format(inventory['stick']), white, font=font)
        draw.text((0, 30), "{}".format(inventory['stone']), white, font=font)
        draw.text((16, 30), "{}".format(inventory['stone_axe']), white, font=font)
        draw.text((32, 30), "{}".format(inventory['stone_pickaxe']), white, font=font)
        draw.text((48, 30), "{}".format(inventory['torch']), white, font=font)
        draw.text((0, 40), "{}".format(inventory['wooden_axe']), white, font=font)
        draw.text((16, 40), "{}".format(inventory['wooden_pickaxe']), white, font=font)
        mainhand = obs['equipped_items']['mainhand']
        draw.text((32, 40), "{}".format(mainhand['damage']), white, font=font)
        draw.text((48, 40), "{}".format(mainhand['maxDamage']), white, font=font)
        draw.text((20, 50), "{}".format(mainhand['type']), white, font=font)

        inv = np.asarray(canvas).astype(np.uint8)[..., np.newaxis]
        frame = obs['pov']
        obs = np.concatenate((frame, inv), axis=2)

        return obs

    def preprocess_states(self, obs):
        result = []
        for i in range(self.config.settings.model.seq_len):
            canvas = Image.new(mode="L", size=(self.input_dim, self.input_dim))
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default()
            white = (255,)

            inventory = obs['inventory']

            draw.text((0, 0), "{}".format(inventory['coal'][i]), white, font=font)
            draw.text((16, 0), "{}".format(inventory['cobblestone'][i]), white, font=font)
            draw.text((32, 0), "{}".format(inventory['crafting_table'][i]), white, font=font)
            draw.text((48, 0), "{}".format(inventory['dirt'][i]), white, font=font)
            draw.text((0, 10), "{}".format(inventory['furnace'][i]), white, font=font)
            draw.text((16, 10), "{}".format(inventory['iron_axe'][i]), white, font=font)
            draw.text((32, 10), "{}".format(inventory['iron_ingot'][i]), white, font=font)
            draw.text((48, 10), "{}".format(inventory['iron_ore'][i]), white, font=font)
            draw.text((0, 20), "{}".format(inventory['iron_pickaxe'][i]), white, font=font)
            draw.text((16, 20), "{}".format(inventory['log'][i]), white, font=font)
            draw.text((32, 20), "{}".format(inventory['planks'][i]), white, font=font)
            draw.text((48, 20), "{}".format(inventory['stick'][i]), white, font=font)
            draw.text((0, 30), "{}".format(inventory['stone'][i]), white, font=font)
            draw.text((16, 30), "{}".format(inventory['stone_axe'][i]), white, font=font)
            draw.text((32, 30), "{}".format(inventory['stone_pickaxe'][i]), white, font=font)
            draw.text((48, 30), "{}".format(inventory['torch'][i]), white, font=font)
            draw.text((0, 40), "{}".format(inventory['wooden_axe'][i]), white, font=font)
            draw.text((16, 40), "{}".format(inventory['wooden_pickaxe'][i]), white, font=font)
            mainhand = obs['equipped_items']['mainhand']
            draw.text((32, 40), "{}".format(mainhand['damage'][i]), white, font=font)
            draw.text((48, 40), "{}".format(mainhand['maxDamage'][i]), white, font=font)
            draw.text((20, 50), "{}".format(mainhand['type'][i]), white, font=font)

            inv = np.asarray(canvas).astype(np.uint8)[..., np.newaxis]
            frame = obs['pov'][i]
            result.append(np.concatenate((frame, inv), axis=2))

        return np.stack(result, axis=0)

    def normalize_state(self, obs):
        return obs.astype(np.float32) / 255.

    def reshape_state(self, obs):
        return np.transpose(obs, (2, 0, 1))

    def reshape_states(self, obs):
        return np.transpose(obs, (0, 3, 1, 2))

    def stack_actions(self, actions):
        result = dict.fromkeys(actions[0].keys(), [])
        for k in result.keys():
            result[k] = []
            for a in actions:
                result[k].append(a[k])
        for k in result.keys():
            result[k] = torch.from_numpy(np.stack(result[k], axis=0)).to(self.device)[:, -1, ...]

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

        del result['attack']
        del result['forward']
        del result['jump']
        del result['back']
        del result['left']
        del result['right']
        del result['sneak']
        del result['sprint']
        return result
