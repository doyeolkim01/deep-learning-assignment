import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedLearning(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.num_features, num_classes)

    def forward(self, batch):
        x, y = batch
        features = self.encoder(x)          # (B, num_features)
        logits = self.classifier(features)  # (B, num_classes)
        loss = F.cross_entropy(logits, y)
        return loss

    # For evaluation(or monitoring)
    @ torch.no_grad()
    def predict(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits
