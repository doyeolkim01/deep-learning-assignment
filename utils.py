import random
import numpy as np
import torch
from methods.rotnet import generate_rotations


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@ torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits = model.predict(x)
        pred = logits.argmax(dim=1)

        total += y.size(0)
        correct += (pred == y).sum().item()

    return correct / total


@ torch.no_grad()
def evaluate_rotnet(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    for x, _ in dataloader:
        x = x.to(device)
        x_rot, y_rot = generate_rotations(x)
        
        logits = model.predict(x)
        
        pred = logits.argmax(dim=1)
        total += y_rot.size(0)
        correct += (pred == y_rot).sum().item()

    return correct / total