import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        # smooth factor
        self.epsilon = epsilon

    def forward(self, logits, targets):
        batch_size = targets.size(0)
        logits = logits.view(batch_size, -1).type(torch.FloatTensor)
        targets = targets.view(batch_size, -1).type(torch.FloatTensor)
        intersection = (logits * targets).sum()
        dice_score = 2. * intersection / ((logits + targets).sum() + self.epsilon)
        return torch.mean(1. - dice_score)


