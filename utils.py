import random
import numpy as np
import torch
import torch.nn.functional as F
from methods.rotnet import generate_rotations


def set_seed(seed=42):
    # set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_device(batch, device):
    # move tensor or nested structure to device(cpu/gpu)
    if torch.is_tensor(batch):
        batch = batch.to(device)
    if isinstance (batch, (list, tuple)):
        return type(batch)(to_device(b, device) for b in batch)
    return batch
        

@ torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits = model.predict(x)    # forward pass
        pred = logits.argmax(dim=1)  # predicted labels

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
        x_rot, y_rot = generate_rotations(x)  # create rotated images and labels
        
        logits = model.predict_rotations(x)   # predict rotation classes
        
        pred = logits.argmax(dim=1)
        total += y_rot.size(0)
        correct += (pred == y_rot).sum().item()

    return correct / total


@ torch.no_grad()
def knn_evaluation(model, valloader, testloader, device):
    model.eval()
    encoder = model.encoder
    
    feature_bank = []
    feature_labels = []
    
    # build feature_bank and feature_labels from validation set
    for x, y in valloader:
        x = x.to(device)
        
        features = encoder(x)                             # (B, D)
        features = F.normalize(features, dim=1)
        
        feature_bank.append(features.cpu())
        feature_labels.append(y)
        
    feature_bank = torch.cat(feature_bank, dim=0)         # (B, D) -> (N, D)
    feature_labels = torch.cat(feature_labels, dim=0)     # (N,)
        
    # apply knn classification on test dataset
    correct = 0
    total = 0
    
    for x, y in testloader:
        x = x.to(device)
        y = y.to(device)
        
        features = encoder(x)                   # (B, D)
        features = F.normalize(features, dim=1)
        
        similarity = features @ feature_bank.T  # compute cosine similarity
        idx = similarity.argmax(dim=1)
        pred = feature_labels[idx].to(device)
        
        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total
        
    
        
        