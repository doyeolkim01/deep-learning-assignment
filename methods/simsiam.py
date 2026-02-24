import torch
import torch.nn as nn
import torch.nn.functional as F

class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, output_dim=2048):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim)
        )
    
    def forward(self, x):
        return self.projector(x)
    
    
class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=2048):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, output_dim)            
        )
    
    def forward(self, x):
        return self.predictor(x)


class SimSiam(nn.Module):
    def __init__(self, encoder, proj_hidden_dim=2048, proj_output_dim=2048, pred_hidden_dim=512):
        super().__init__()
        self.encoder = encoder
        feat_dim = encoder.num_features
        
        self.projector = Projector(input_dim=feat_dim, hidden_dim=proj_hidden_dim, output_dim=proj_output_dim)
        self.predictor = Predictor(input_dim=proj_output_dim, hidden_dim=pred_hidden_dim, output_dim=proj_output_dim)
    
    # compute negative cosine similarity    
    def neg_cos_sim(self, p, z):
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        
        return -(p * z).sum(dim=1).mean()
    
    def forward(self, batch):
        # get two augmented views of the same batch from "datasets.py"
        (x1, x2), _ = batch
        
        # Encoder(Result: (b, D))
        h1 = self.encoder(x1) 
        h2 = self.encoder(x2)
        
        # Projector(Result: (b, d))
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        
        # Predictor(Result: (b, d))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        # SimSiam loss with stop gradient
        loss = 0.5 * self.neg_cos_sim(p1, z2.detach()) + 0.5 * self.neg_cos_sim(p2, z1.detach())
        return loss
        
    
    
        
        