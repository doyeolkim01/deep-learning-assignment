import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# MLP used for both projector and predictor
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=4096, output_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, output_dim, bias=False)
        )
    
    def forward(self, x):
        return self.mlp(x)


class BYOL(nn.Module):
    def __init__(self, encoder, hidden_dim=4096, output_dim=256, momentum = 0.99):
        super().__init__()
        self.momentum = momentum
        self.q_encoder = encoder
        self.k_encoder = copy.deepcopy(encoder) # create key encoder by copying query encoder, using different weights
        feat_dim = encoder.num_features
        
        self.q_projector = MLP(input_dim=feat_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.k_projector = copy.deepcopy(self.q_projector)
        
        self.q_predictor = MLP(input_dim=output_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        
        for p in self.k_encoder.parameters():
            p.requires_grad = False
        for p in self.k_projector.parameters():
            p.requires_grad = False
   
    
    @torch.no_grad()
    def momentum_update(self):
        # "θ_k" is replaced with "momentum · θ_k + (1 − momentum) · θ_q"
        # In-place multiplication and addition are used 
        for q, k in zip(self.q_encoder.parameters(), self.k_encoder.parameters()):
            k.data.mul_(self.momentum).add_(q.data, alpha=1-self.momentum)
        
        for q, k in zip(self.q_projector.parameters(), self.k_projector.parameters()):
            k.data.mul_(self.momentum).add_(q.data, alpha=1-self.momentum)
    
    
    # compute negative cosine similarity    
    def neg_cos_sim(self, p, z):
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        
        return -(p * z).sum(dim=1).mean()
    
    
    def forward(self, batch):
        # get two augmented views of the same batch from "datasets.py"
        (x1, x2), _ = batch
        
        # Online
        h1_q = self.q_encoder(x1) 
        h2_q = self.q_encoder(x2)
        z1_q = self.q_projector(h1_q)
        z2_q = self.q_projector(h2_q)
        p1_q = self.q_predictor(z1_q)
        p2_q = self.q_predictor(z2_q)
        
        # Target
        with torch.no_grad():
            h1_k = self.k_encoder(x1) 
            h2_k = self.k_encoder(x2)
            z1_k = self.k_projector(h1_k)
            z2_k = self.k_projector(h2_k)
        
        # SimSiam loss with stop gradient
        loss = 0.5 * self.neg_cos_sim(p1_q, z2_k) + 0.5 * self.neg_cos_sim(p2_q, z1_k)
        return loss
    
    # Use q_encoder as the main encoder 
    @property
    def encoder(self):
        return self.q_encoder
        
    
    
        
        