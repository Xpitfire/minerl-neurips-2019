import torch
from lighter.decorator import context


class BehaviouralCloningLoss(torch.nn.Module):
    @context
    def __init__(self):
        super(BehaviouralCloningLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss()
        self.bce = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, target):
        camera1 = target['actions']['camera1'].long()
        camera2 = target['actions']['camera2'].long()
        camera_loss = self.ce(pred['camera1_net_out'], camera1) + self.ce(pred['camera2_net_out'], camera2)
        move_loss = 0.
        for m in range(self.config.settings.model.actor_move_final_dim):
            move = target['actions']['move'][:, m].float()
            move_loss = move_loss + self.bce(pred['move_net_out'][:, m], move)
        craft = target['actions']['craft'].long()
        craft_loss = self.ce(pred['craft_net_out'], craft)
        equip = target['actions']['equip'].long()
        equip_loss = self.ce(pred['equip_net_out'], equip)
        place = target['actions']['place'].long()
        place_loss = self.ce(pred['place_net_out'], place)
        nearbyCraft = target['actions']['nearbyCraft'].long()
        nearbyCraft_loss = self.ce(pred['nearbyCraft_net_out'], nearbyCraft)
        nearbySmelt = target['actions']['nearbySmelt'].long()
        nearbySmelt_loss = self.ce(pred['nearbySmelt_net_out'], nearbySmelt)
        # log rescaling trick
        value_loss = torch.log(1 + self.mse(pred['value'], target['values']))
        loss = camera_loss + move_loss + craft_loss + equip_loss + place_loss + nearbyCraft_loss + nearbySmelt_loss + value_loss
        return loss
