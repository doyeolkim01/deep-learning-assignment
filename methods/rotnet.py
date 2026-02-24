import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import generate_rotations


class RotNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.num_features, 4)

    def forward(self, batch):
        x, _ = batch                            # Original labels are not used! (self-supervised pretext task)
        x_rot, y_rot = generate_rotations(x)
        
        features = self.encoder(x_rot)          # (4B, num_features)
        logits = self.classifier(features)      # (4B, 4)
        loss = F.cross_entropy(logits, y_rot)
        return loss

    # For evaluation(or monitoring)
    @ torch.no_grad()
    def predict_rotations(self, x):
        x_rot, _ = generate_rotations(x)
        features = self.encoder(x_rot)
        logits = self.classifier(features)
        return logits