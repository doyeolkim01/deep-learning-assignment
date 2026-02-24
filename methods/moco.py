import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Projection Head for Moco (v1:Linear, v2: MLP)
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=128, version='v2'):
        super().__init__()
        
        if version == 'v2':
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)            
            )
            
        else:
            self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.projection(x)


class MoCo(nn.Module):
    def __init__(self, encoder, version='v2', proj_hidden_dim=512, proj_output_dim=128, 
                 queue_size=4096, momentum=0.999, temperature=0.2):
        super().__init__()
        assert version in ["v1", "v2"]
        
        self.version = version
        self.queue_size = int(queue_size)
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        
        self.q_encoder = encoder
        self.k_encoder = copy.deepcopy(encoder) # create key encoder by copying query encoder, using different weights
        
        input_dim = encoder.num_features
        
        self.q_head = ProjectionHead(input_dim=input_dim, hidden_dim=proj_hidden_dim, output_dim=proj_output_dim, version=version)
        self.k_head = ProjectionHead(input_dim=input_dim, hidden_dim=proj_hidden_dim, output_dim=proj_output_dim, version=version)
        
        # no gradient update in key encoder & head
        for p in list(self.k_encoder.parameters()) + list(self.k_head.parameters()):
            p.requires_grad = False
            
        # queue: stores embedded negative keys, shape: (proj_output_dim, queue_size), not updated
        self.register_buffer("queue", F.normalize(torch.randn(proj_output_dim, self.queue_size), dim=0))
            
        # queue pointer: indicates the current position in the queue (FIFO behavior), not updated
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def momentum_update(self):
        # "θ_k" is replaced with "momentum · θ_k + (1 − momentum) · θ_q"
        # In-place multiplication and addition are used 
        for q, k in zip(self.q_encoder.parameters(), self.k_encoder.parameters()):
            k.data.mul_(self.momentum).add_(q.data, alpha=1-self.momentum)
        
        for q, k in zip(self.q_head.parameters(), self.k_head.parameters()):
            k.data.mul_(self.momentum).add_(q.data, alpha=1-self.momentum)
    
    @torch.no_grad()
    def _enqueue(self, keys):
        # keys are normalized with shape (B, D)
        batch_size = keys.size(0)
        q_ptr = int(self.queue_ptr.item())
        end = batch_size + q_ptr
        
        if end <= self.queue_size:
            self.queue[:, q_ptr:end] = keys.T
        else:
            left = self.queue_size - q_ptr
            self.queue[:, q_ptr:] = keys[:left].T
            self.queue[:, :(end - self.queue_size)] = keys[left:].T
        
        self.queue_ptr[0] = end % self.queue_size
         
    
    def forward(self, batch):
        (x1, x2), _ = batch
        
        # query encoder + head
        q = self.q_encoder(x1)
        q = self.q_head(q)
        q = F.normalize(q, dim=1)
        
        # key encoder(momentum encoder) + head
        with torch.no_grad():
            k = self.k_encoder(x2)
            k = self.k_head(k)
            k = F.normalize(k, dim=1)
        
        # compute contrastive loss
        positive_logits = (q * k).sum(dim=1, keepdim=True)              # (B, 1)
        negative_logits = q @ self.queue.clone().detach()               # (B, queue_size)
        logits = torch.cat([positive_logits, negative_logits], dim=1)   # (B, 1+queue_size)
        logits /= self.temperature
        
        # Answer labels are always at index 0(postiive logits)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        
        # update queue
        with torch.no_grad():
            self._enqueue(k)
        
        return loss
    
    # Use q_encoder as the main encoder 
    @property
    def encoder(self):
        return self.q_encoder
        
        
        
        
        
            

        
        
        
        