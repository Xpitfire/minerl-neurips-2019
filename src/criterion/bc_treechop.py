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
        assert self.config.settings.model.actor_camera_final_dim == pred['camera1_net_out'].shape[-1]
        assert self.config.settings.model.actor_camera_final_dim == pred['camera2_net_out'].shape[-1]
        assert self.config.settings.model.actor_move_final_dim == pred['move_net_out'].shape[-1]

        camera1 = target['actions']['camera1'].long()
        camera2 = target['actions']['camera2'].long()
        camera_loss = self.ce(pred['camera1_net_out'], camera1) + self.ce(pred['camera2_net_out'], camera2)
        move_loss = 0.
        for m in range(self.config.settings.model.actor_move_final_dim):
            move = target['actions']['move'][:, m].float()
            move_loss = move_loss + self.bce(pred['move_net_out'][:, m], move)

        # log rescaling trick
        value_loss = torch.log(1 + self.mse(pred['value'], target['values']))
        loss = camera_loss + move_loss + value_loss
        return loss
