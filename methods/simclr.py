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
        x1, x2 = batch
        
        # Encoder(Result: (B,D))
        h1 = self.encoder(x1) 
        h2 = self.encoder(x2)
        
        # Projection head(Result: (B,d))
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
    
        logits = (z @ z.t()) / self.temperature # (2b, 2b)
        logits.fill_diagonal_(1e-9)
    
        # positive pairs: (i, i+b) and (i+b, i) (mod 2b)
        idx = torch.arange(2*b, device=z.device)
        pos_idx = (idx+b) % (2*b)
    
        pos = logits[idx, pos_idx] # extract (0,b), (1,b+1), ... , (b,0), (b+1,2), ... from logits, shape: (2b,)
    
        loss = (-pos + torch.logsumexp(logits, dim=1)).mean()
        return loss
    
    
        
        