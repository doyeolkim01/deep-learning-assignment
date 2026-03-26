import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class DINOHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, bottleneck_dim=256, output_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        # weight를 방향(v)과 크기(g)로 분리하는 weight normalization 적용
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, output_dim, bias=False))
        # weight의 scale(g)을 1로 초기화하여 unit vector 상태로 시작
        self.last_layer.weight_g.data.fill_(1.0)
        # scale(g) 학습을 막아 방향만 학습하도록 함
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class DINOLoss(nn.Module):
    def __init__(self, output_dim, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        
        self.register_buffer("center", torch.zeros(1, output_dim))
    
    def forward(self, student_outputs, teacher_outputs):
        total_loss = 0.0
        n_loss_terms = 0
        
        for t, t_out in enumerate(teacher_outputs):
            t_prob = F.softmax((t_out - self.center) / self.teacher_temp, dim=-1).detach()
            
            for s, s_out in enumerate(student_outputs):
                # skip same-view!
                if s == t:
                    continue
                
                s_log_prob = F.log_softmax(s_out / self.student_temp, dim = -1)
                loss = torch.sum(-t_prob * s_log_prob, dim = -1).mean()
                total_loss += loss
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        self.update_center(teacher_outputs)
        
        return total_loss
    
    @torch.no_grad()
    def update_center(self, teacher_outputs):
        batch_center = torch.cat(teacher_outputs, dim=0).mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)
            

class DINO(nn.Module):
    def __init__(self, encoder, hidden_dim=512, bottleneck_dim=256, output_dim=1024,
                 student_temp=0.1, teacher_temp=0.04, center_momentum=0.9, teacher_momentum=0.996):
        super().__init__()
        
        # student와 teacher pipeline 분리 
        self.student_encoder = encoder
        self.student_head = DINOHead(input_dim=encoder.num_features, hidden_dim=hidden_dim,
                                     bottleneck_dim=bottleneck_dim, output_dim=output_dim)
        
        self.teacher_encoder = copy.deepcopy(encoder)
        self.teacher_head = DINOHead(input_dim=encoder.num_features, hidden_dim=hidden_dim,
                                     bottleneck_dim=bottleneck_dim, output_dim=output_dim)
        
        # teaccher를 student와 동일하게 초기화
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        
        self.teacher_momentum = teacher_momentum
        
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False
            
        self.dino_loss = DINOLoss(output_dim=output_dim, student_temp=student_temp,
                                  teacher_temp=teacher_temp, center_momentum=center_momentum)
        
    
    def forward(self, batch):
        """
        batch: views, _
        views = [global_view1, global_view2, local_view1, local_view2, ...]
        """
        views, _ = batch
        global_views = views[:2]
        
        student_outputs = []
        for v in views:
            feat = self.student_encoder(v)
            out = self.student_head(feat)
            student_outputs.append(out)
        
        teacher_outputs = []
        with torch.no_grad():
            for v in global_views:
                feat = self.teacher_encoder(v)
                out = self.teacher_head(feat)
                teacher_outputs.append(out)
        
        loss = self.dino_loss(student_outputs, teacher_outputs)
        return loss
    
    @torch.no_grad()
    def update_teacher(self):
        for student_param, teacher_param in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            teacher_param.data = teacher_param.data * self.teacher_momentum + student_param.data * (1 - self.teacher_momentum)
            
        for student_param, teacher_param in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            teacher_param.data = teacher_param.data * self.teacher_momentum + student_param.data * (1 - self.teacher_momentum)
