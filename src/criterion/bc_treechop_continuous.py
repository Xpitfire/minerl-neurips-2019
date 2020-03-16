import torch
from lighter.decorator import context, reference


class BehaviouralCloningLoss(torch.nn.Module):
    @context
    @reference(name='collectible')
    def __init__(self):
        super(BehaviouralCloningLoss, self).__init__()
        self.bce = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, target):
        assert self.config.settings.model.actor_camera_final_dim == pred['camera1_logits'].shape[-1]
        assert self.config.settings.model.actor_camera_final_dim == pred['camera2_logits'].shape[-1]
        assert self.config.settings.model.actor_move_final_dim == pred['move_logits'].shape[-1]

        camera1 = target['actions']['camera1'].view(-1, 1)
        camera2 = target['actions']['camera2'].view(-1, 1)
        camera_loss = torch.log(1 + self.mse(pred['camera1_logits'], camera1) + self.mse(pred['camera2_logits'], camera2))
        move_loss = 0.
        for m in range(self.config.settings.model.actor_move_final_dim):
            move = target['actions']['move'][:, m].float()
            move_loss = move_loss + self.bce(torch.sigmoid(pred['move_logits'][:, m]), move)

        # log rescaling trick
        value_loss = torch.log(1 + self.mse(pred['value'], target['values']))
        loss = camera_loss + move_loss + value_loss
        self.collectible.update(category='train', **{'loss': loss.detach().cpu().item()})
        self.collectible.update(category='train', **{'value_loss': value_loss.detach().cpu().item()})
        self.collectible.update(category='train', **{'move_loss': move_loss.detach().cpu().item()})
        self.collectible.update(category='train', **{'camera_loss': camera_loss.detach().cpu().item()})
        return loss
