import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_rotations(x):
    '''
    x: (B, C, H, W)
    x_rot = (B, C, H, W) x 4 = (4B, C, H, W)
    y_rot = (B,) x 4 = (4B,) for labels 0(0 degree), 1(90 degree), 2(180 degree), 3(270 degree)
    '''
    x0 = x
    x1 = torch.rot90(x, k=1, dims=(2,3))
    x2 = torch.rot90(x, k=2, dims=(2,3))
    x3 = torch.rot90(x, k=3, dims=(2,3))
    x_rot = torch.cat([x0, x1, x2, x3], dim=0)
    
    b = x.size(0)
    
    y0 = torch.zeros(b, device = x.device, dtype = torch.long)
    y1 = torch.ones(b, device = x.device, dtype = torch.long)
    y2 = torch.full((b,), 2, device = x.device, dtype = torch.long)
    y3 = torch.full((b,), 3, device = x.device, dtype = torch.long)
    y_rot = torch.cat([y0, y1, y2, y3], dim=0)
    
    return x_rot, y_rot


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
    def predict(self, x):
        x_rot, _ = generate_rotations(x)
        features = self.encoder(x_rot)
        logits = self.classifier(features)
        return logits