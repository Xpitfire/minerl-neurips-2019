import torch


class BehaviouralCloningLoss(torch.nn.Module):
    def __init__(self):
        super(BehaviouralCloningLoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, batch, pred, evaluation):
        camera_loss = torch.log(1 + self.mse(pred['camera_log_probs'], evaluation['camera_log_probs']))
        move_loss = torch.log(1 + self.mse(pred['move_log_probs'], evaluation['move_log_probs']))
        craft_loss = torch.log(1 + self.mse(pred['craft_log_probs'], evaluation['craft_log_probs']))
        equip_loss = torch.log(1 + self.mse(pred['equip_log_probs'], evaluation['equip_log_probs']))
        place_loss = torch.log(1 + self.mse(pred['place_log_probs'], evaluation['place_log_probs']))
        nearbyCraft_loss = torch.log(1 + self.mse(pred['nearbyCraft_log_probs'], evaluation['nearbyCraft_log_probs']))
        nearbySmelt_loss = torch.log(1 + self.mse(pred['nearbySmelt_log_probs'], evaluation['nearbySmelt_log_probs']))
        value_loss = torch.log(1 + self.mse(evaluation['values'], batch['values']))
        loss = camera_loss + move_loss + craft_loss + equip_loss + place_loss + nearbyCraft_loss + nearbySmelt_loss + value_loss
        return loss
