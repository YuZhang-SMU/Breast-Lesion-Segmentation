import torch
import torch.nn as nn

class Dice_score(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(Dice_score, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        batch_size = logits.size(0)
        m1 = logits.view(batch_size, -1)
        m1 = torch.where(m1 >= 0.5, 1.0, 0.0)
        m2 = targets.view(batch_size, -1)
        intersection = (m1 * m2).sum()
        return (2. * intersection + self.epsilon) / (m1.sum() + m2.sum() + self.epsilon)