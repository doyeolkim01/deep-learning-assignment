import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)            
        )
    
    def forward(self, x):
        return self.projection(x)


class SimCLR(nn.Module):
    def __init__(self, encoder, temperature=0.5, proj_hidden_dim=512, proj_output_dim=128):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(input_dim = encoder.num_features, hidden_dim = proj_hidden_dim, output_dim = proj_output_dim)
        self.temperature = temperature
    
    def forward(self, batch):
        # get two augmented views of the same batch from "datasets.py"
        (x1, x2), _ = batch
        
        # Encoder(Result: (b, D))
        h1 = self.encoder(x1) 
        h2 = self.encoder(x2)
        
        # Projection head(Result: (b, d))
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        
        # L2 normalization
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        loss = self.nt_xent_loss(z1, z2)
        return loss
        
    def nt_xent_loss(self, z1, z2):
        b = z1.size(0)
        z = torch.cat([z1, z2], dim=0) # (2b, d)
    
        # cosine simliarity logits
        logits = (z @ z.t()) / self.temperature # (2b, 2b)
        
        # remove self-similarity
        mask = torch.eye(2*b, device = z.device, dtype=torch.bool)
        logits = logits.masked_fill(mask, -1e9)
    
        # positive pairs: (i, i+b) and (i+b, i) (mod 2b)
        targets = torch.arange(2 * b, device=z.device)
        targets = (targets + b) % (2 * b) # (2b,)
    
        # cross-entropy over 2b candidates
        loss = F.cross_entropy(logits, targets)
        return loss
    
    
        
        