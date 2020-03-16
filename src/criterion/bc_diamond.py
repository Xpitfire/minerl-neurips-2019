import torch
from lighter.decorator import context, reference


class BehaviouralCloningLoss(torch.nn.Module):
    @context
    @reference(name='collectible')
    def __init__(self):
        super(BehaviouralCloningLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss()
        self.bce = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, target):
        assert self.config.settings.model.actor_camera_final_dim == pred['camera1_logits'].shape[-1]
        assert self.config.settings.model.actor_camera_final_dim == pred['camera2_logits'].shape[-1]
        assert self.config.settings.model.actor_move_final_dim == pred['move_logits'].shape[-1]
        assert self.config.settings.model.actor_craft_final_dim == pred['craft_logits'].shape[-1]
        assert self.config.settings.model.actor_equip_final_dim == pred['equip_logits'].shape[-1]
        assert self.config.settings.model.actor_place_final_dim == pred['place_logits'].shape[-1]
        assert self.config.settings.model.actor_nearbyCraft_final_dim == pred['nearbyCraft_logits'].shape[-1]
        assert self.config.settings.model.actor_nearbySmelt_final_dim == pred['nearbySmelt_logits'].shape[-1]

        camera1 = target['actions']['camera1'].long()
        camera2 = target['actions']['camera2'].long()
        camera_loss = self.ce(pred['camera1_logits'], camera1) + self.ce(pred['camera2_logits'], camera2)
        move_loss = 0.
        for m in range(self.config.settings.model.actor_move_final_dim):
            move = target['actions']['move'][:, m].float()
            move_loss = move_loss + self.bce(pred['move_logits'][:, m], move)
        craft = target['actions']['craft'].long()
        craft_loss = self.ce(pred['craft_logits'], craft)
        equip = target['actions']['equip'].long()
        equip_loss = self.ce(pred['equip_logits'], equip)
        place = target['actions']['place'].long()
        place_loss = self.ce(pred['place_logits'], place)
        nearbyCraft = target['actions']['nearbyCraft'].long()
        nearbyCraft_loss = self.ce(pred['nearbyCraft_logits'], nearbyCraft)
        nearbySmelt = target['actions']['nearbySmelt'].long()
        nearbySmelt_loss = self.ce(pred['nearbySmelt_logits'], nearbySmelt)
        # log rescaling trick
        value_loss = torch.log(1 + self.mse(pred['value'], target['values']))
        loss = camera_loss + move_loss + craft_loss + equip_loss + place_loss + nearbyCraft_loss + nearbySmelt_loss + value_loss
        self.collectible.update(category='train', **{'loss': loss.detach().cpu().item()})
        self.collectible.update(category='train', **{'value_loss': value_loss.detach().cpu().item()})
        self.collectible.update(category='train', **{'move_loss': move_loss.detach().cpu().item()})
        self.collectible.update(category='train', **{'camera_loss': camera_loss.detach().cpu().item()})
        self.collectible.update(category='train', **{'craft_loss': craft_loss.detach().cpu().item()})
        self.collectible.update(category='train', **{'equip_loss': equip_loss.detach().cpu().item()})
        self.collectible.update(category='train', **{'place_loss': place_loss.detach().cpu().item()})
        self.collectible.update(category='train', **{'nearbyCraft_loss': nearbyCraft_loss.detach().cpu().item()})
        self.collectible.update(category='train', **{'nearbySmelt_loss': nearbySmelt_loss.detach().cpu().item()})
        return loss
