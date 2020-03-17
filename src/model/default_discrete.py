import torch
import os
import torch.nn as nn
from lighter.nn import DynLayerNorm
from lighter.decorator import context, device
from torch.distributions import Categorical, Bernoulli
from datetime import datetime


class ConvNet(nn.Module):
    @context
    def __init__(self):
        super(ConvNet, self).__init__()
        in_channels = self.config.settings.model.input_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=0),
            DynLayerNorm(),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            DynLayerNorm(),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),
            DynLayerNorm(),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            DynLayerNorm(),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

    def forward(self, x):
        h = self.conv(x)
        return h


class CnnLstmNet(nn.Module):
    @context
    @device
    def __init__(self):
        super(CnnLstmNet, self).__init__()
        self.cnn = ConvNet()
        self.in_dim = self.get_dimensions()
        self.hidden_dim = self.config.settings.model.lstm_hidden_dim
        self.layer_dim = self.config.settings.model.lstm_layer_dim
        self.lstm = nn.LSTM(input_size=self.in_dim, hidden_size=self.hidden_dim, num_layers=self.layer_dim,
                            batch_first=True, dropout=0.0)

    def get_dimensions(self):
        input_size = (1, self.config.settings.model.input_channels,
                      self.config.settings.model.input_dim,
                      self.config.settings.model.input_dim)
        dummy = torch.randn(input_size)
        out = self.cnn(dummy)
        return out.shape[-1]*out.shape[-2]*out.shape[-3]

    def forward(self, x):
        b, s, c, h, w = x.shape

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.shape[0], self.hidden_dim,
                         dtype=x.dtype, layout=x.layout, device=x.device).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.shape[0], self.hidden_dim,
                         dtype=x.dtype, layout=x.layout, device=x.device).requires_grad_()

        # reshape time to batch dimension
        h = torch.reshape(x, shape=(b * s, c, h, w))
        # apply cnn encoder
        h = self.cnn(h)
        # split time and batch dimension
        h = torch.reshape(h, shape=(b, s, -1))

        h = self.lstm(h, (h0.detach(), c0.detach()))[0][:, -1, ...]
        return h


class ActorNet(nn.Module):
    @context
    def __init__(self):
        super(ActorNet, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(self.config.settings.model.lstm_hidden_dim, self.config.settings.model.actor_hidden_dim),
            DynLayerNorm(),
            nn.ReLU()
        )
        self.camera_final_dim = self.config.settings.model.actor_camera_final_dim
        self.camera1 = nn.Linear(self.config.settings.model.actor_hidden_dim,
                                 self.camera_final_dim)
        self.camera2 = nn.Linear(self.config.settings.model.actor_hidden_dim,
                                 self.camera_final_dim)
        self.move = nn.Linear(self.config.settings.model.actor_hidden_dim,
                              self.config.settings.model.actor_move_final_dim)
        self.craft = nn.Linear(self.config.settings.model.actor_hidden_dim,
                               self.config.settings.model.actor_craft_final_dim)
        self.equip = nn.Linear(self.config.settings.model.actor_hidden_dim,
                               self.config.settings.model.actor_equip_final_dim)
        self.place = nn.Linear(self.config.settings.model.actor_hidden_dim,
                               self.config.settings.model.actor_place_final_dim)
        self.nearbyCraft = nn.Linear(self.config.settings.model.actor_hidden_dim,
                                     self.config.settings.model.actor_nearbyCraft_final_dim)
        self.nearbySmelt = nn.Linear(self.config.settings.model.actor_hidden_dim,
                                     self.config.settings.model.actor_nearbySmelt_final_dim)

    def forward(self, x):
        h = self.linear(x)

        camera1_logits = self.camera1(h)
        camera2_logits = self.camera2(h)
        move_logits = self.move(h)
        craft_logits = self.craft(h)
        equip_logits = self.equip(h)
        place_logits = self.place(h)
        nearbyCraft_logits = self.nearbyCraft(h)
        nearbySmelt_logits = self.nearbySmelt(h)

        return {
            'camera1_logits': camera1_logits,
            'camera2_logits': camera2_logits,
            'move_logits': move_logits,
            'craft_logits': craft_logits,
            'equip_logits': equip_logits,
            'place_logits': place_logits,
            'nearbyCraft_logits': nearbyCraft_logits,
            'nearbySmelt_logits': nearbySmelt_logits
        }


class CriticNet(nn.Module):
    @context
    def __init__(self):
        super(CriticNet, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(self.config.settings.model.lstm_hidden_dim, self.config.settings.model.critic_hidden_dim),
            DynLayerNorm(),
            nn.ReLU()
        )
        self.value = nn.Linear(self.config.settings.model.critic_hidden_dim,
                               self.config.settings.model.critic_final_dim)

    def forward(self, x):
        h = self.linear(x)
        return self.value(h)


class Net(nn.Module):
    @device
    @context
    def __init__(self):
        super(Net, self).__init__()
        self.checkpoints_dir = self.config.settings.model.checkpoints_dir
        self.shared = CnnLstmNet()
        self.actor = ActorNet()
        self.critic = CriticNet()

    def act(self, state):
        h = self.shared(state)
        logits_dict = self.actor(h)
        value = self.critic(h)

        camera1_probs = torch.softmax(logits_dict['camera1_logits'], dim=-1)
        camera1_dist = Categorical(probs=camera1_probs)
        camera1_action = camera1_dist.sample()
        camera1_log_probs = camera1_dist.log_prob(camera1_action)

        camera2_probs = torch.softmax(logits_dict['camera2_logits'], dim=-1)
        camera2_dist = Categorical(probs=camera2_probs)
        camera2_action = camera2_dist.sample()
        camera2_log_probs = camera2_dist.log_prob(camera2_action)

        move_probs = torch.sigmoid(logits_dict['move_logits'])
        move_dist = Bernoulli(probs=move_probs)
        move_action = move_dist.sample()
        move_log_probs = move_dist.log_prob(move_action)

        craft_probs = torch.softmax(logits_dict['craft_logits'], dim=-1)
        craft_dist = Categorical(probs=craft_probs)
        craft_action = craft_dist.sample()
        craft_log_probs = craft_dist.log_prob(craft_action)

        equip_probs = torch.softmax(logits_dict['equip_logits'], dim=-1)
        equip_dist = Categorical(probs=equip_probs)
        equip_action = equip_dist.sample()
        equip_log_probs = equip_dist.log_prob(equip_action)

        place_probs = torch.softmax(logits_dict['place_logits'], dim=-1)
        place_dist = Categorical(probs=place_probs)
        place_action = place_dist.sample()
        place_log_probs = place_dist.log_prob(place_action)

        nearbyCraft_probs = torch.softmax(logits_dict['nearbyCraft_logits'], dim=-1)
        nearbyCraft_dist = Categorical(probs=nearbyCraft_probs)
        nearbyCraft_action = nearbyCraft_dist.sample()
        nearbyCraft_log_probs = nearbyCraft_dist.log_prob(nearbyCraft_action)

        nearbySmelt_probs = torch.softmax(logits_dict['nearbySmelt_logits'], dim=-1)
        nearbySmelt_dist = Categorical(probs=nearbySmelt_probs)
        nearbySmelt_action = nearbySmelt_dist.sample()
        nearbySmelt_log_probs = nearbySmelt_dist.log_prob(nearbySmelt_action)

        return {
            'camera1': camera1_action,
            'camera1_log_probs': camera1_log_probs,
            'camera1_logits': logits_dict['camera1_logits'],
            'camera2': camera2_action,
            'camera2_log_probs': camera2_log_probs,
            'camera2_logits': logits_dict['camera2_logits'],
            'move': move_action,
            'move_log_probs': move_log_probs,
            'move_logits': logits_dict['move_logits'],
            'craft': craft_action,
            'craft_log_probs': craft_log_probs,
            'craft_logits': logits_dict['craft_logits'],
            'equip': equip_action,
            'equip_log_probs': equip_log_probs,
            'equip_logits': logits_dict['equip_logits'],
            'place': place_action,
            'place_log_probs': place_log_probs,
            'place_logits': logits_dict['place_logits'],
            'nearbyCraft': nearbyCraft_action,
            'nearbyCraft_log_probs': nearbyCraft_log_probs,
            'nearbyCraft_logits': logits_dict['nearbyCraft_logits'],
            'nearbySmelt': nearbySmelt_action,
            'nearbySmelt_log_probs': nearbySmelt_log_probs,
            'nearbySmelt_logits': logits_dict['nearbySmelt_logits'],
            'value': value
        }

    def evaluate(self, state, action):
        h = self.shared(state)
        logits_dict = self.actor(h)
        value = self.critic(h)

        camera1_probs = torch.softmax(logits_dict['camera1_logits'], dim=-1)
        camera1_dist = Categorical(probs=camera1_probs)
        camera1_log_probs = camera1_dist.log_prob(action['camera1'])
        camera1_entropy = camera1_dist.entropy()

        camera2_probs = torch.softmax(logits_dict['camera2_logits'], dim=-1)
        camera2_dist = Categorical(probs=camera2_probs)
        camera2_log_probs = camera2_dist.log_prob(action['camera2'])
        camera2_entropy = camera2_dist.entropy()

        move_probs = torch.sigmoid(logits_dict['move_logits'])
        move_dist = Bernoulli(probs=move_probs)
        move_log_probs = move_dist.log_prob(action['move'])
        move_entropy = move_dist.entropy()

        craft_probs = torch.softmax(logits_dict['craft_logits'], dim=-1)
        craft_dist = Categorical(probs=craft_probs)
        craft_log_probs = craft_dist.log_prob(action['craft'])
        craft_entropy = craft_dist.entropy()

        equip_probs = torch.softmax(logits_dict['equip_logits'], dim=-1)
        equip_dist = Categorical(probs=equip_probs)
        equip_log_probs = equip_dist.log_prob(action['equip'])
        equip_entropy = equip_dist.entropy()

        place_probs = torch.softmax(logits_dict['place_logits'], dim=-1)
        place_dist = Categorical(probs=place_probs)
        place_log_probs = place_dist.log_prob(action['place'])
        place_entropy = place_dist.entropy()

        nearbyCraft_probs = torch.softmax(logits_dict['nearbyCraft_logits'], dim=-1)
        nearbyCraft_dist = Categorical(probs=nearbyCraft_probs)
        nearbyCraft_log_probs = nearbyCraft_dist.log_prob(action['nearbyCraft'])
        nearbyCraft_entropy = nearbyCraft_dist.entropy()

        nearbySmelt_probs = torch.softmax(logits_dict['nearbySmelt_logits'], dim=-1)
        nearbySmelt_dist = Categorical(probs=nearbySmelt_probs)
        nearbySmelt_log_probs = nearbySmelt_dist.log_prob(action['nearbySmelt'])
        nearbySmelt_entropy = nearbySmelt_dist.entropy()

        return {
            'camera1': action['camera1'],
            'camera1_log_probs': camera1_log_probs,
            'camera1_entropy': camera1_entropy,
            'camera1_logits': logits_dict['camera1_logits'],
            'camera2': action['camera2'],
            'camera2_log_probs': camera2_log_probs,
            'camera2_entropy': camera2_entropy,
            'camera2_logits': logits_dict['camera2_logits'],
            'move': action['move'],
            'move_log_probs': move_log_probs,
            'move_entropy': move_entropy,
            'move_logits': logits_dict['move_logits'],
            'craft': action['craft'],
            'craft_log_probs': craft_log_probs,
            'craft_entropy': craft_entropy,
            'craft_logits': logits_dict['craft_logits'],
            'equip': action['equip'],
            'equip_log_probs': equip_log_probs,
            'equip_entropy': equip_entropy,
            'equip_logits': logits_dict['equip_logits'],
            'place': action['place'],
            'place_log_probs': place_log_probs,
            'place_entropy': place_entropy,
            'place_logits': logits_dict['place_logits'],
            'nearbyCraft': action['nearbyCraft'],
            'nearbyCraft_log_probs': nearbyCraft_log_probs,
            'nearbyCraft_entropy': nearbyCraft_entropy,
            'nearbyCraft_logits': logits_dict['nearbyCraft_logits'],
            'nearbySmelt': action['nearbySmelt'],
            'nearbySmelt_log_probs': nearbySmelt_log_probs,
            'nearbySmelt_entropy': nearbySmelt_entropy,
            'nearbySmelt_logits': logits_dict['nearbySmelt_logits'],
            'value': value
        }

    def forward(self, x):
        raise NotImplementedError

    def checkpoint(self, epoch: int):
        timestamp = datetime.timestamp(datetime.now())
        file_name = 'e-{}_time-{}'.format(epoch, timestamp)
        path = os.path.join(self.checkpoints_dir, self.config.context_id)
        if not os.path.exists(path):
            os.makedirs(path)
        ckpt_file = os.path.join(path, '{}.ckpt'.format(file_name))
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict()
        }, ckpt_file)

    def load(self, path):
        if path is None or path == "":
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
