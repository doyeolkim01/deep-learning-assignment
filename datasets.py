import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# for RotNet
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
    
    # rotation labels
    y0 = torch.zeros(b, device = x.device, dtype = torch.long)
    y1 = torch.ones(b, device = x.device, dtype = torch.long)
    y2 = torch.full((b,), 2, device = x.device, dtype = torch.long)
    y3 = torch.full((b,), 3, device = x.device, dtype = torch.long)
    y_rot = torch.cat([y0, y1, y2, y3], dim=0)
    
    return x_rot, y_rot


# for SimCLR (return two augmented views)
class TwoTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, x):
        x1 = self.base_transform(x)
        x2 = self.base_transform(x)
        
        return x1, x2


def get_transforms(img_size=32):
    # supervised / rotnet transform
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # validation / test transform
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # ssl transform v1 for MoCo-v1
    ssl_transform_v1 = TwoTransform(transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
    
    
    # ssl transform v2 for SimCLR, MoCo-v2 
    ssl_transform_v2 = TwoTransform(transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale = (0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p = 0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p = 0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
    
    return train_transform, test_transform, ssl_transform_v1, ssl_transform_v2


def get_dataloaders(method_name, batch_size, img_size=32, val_ratio=0.1, root = './data', seed = 42):
    train_transform, test_transform, ssl_transform_v1, ssl_transform_v2 = get_transforms(img_size)
    
    if method_name in ['supervised_learning', 'rotnet']:
        selected_train_transform = train_transform
    elif method_name in ['moco_v1']:
        selected_train_transform = ssl_transform_v1
    elif method_name in ['simclr', 'moco', 'moco_v2']:
        selected_train_transform = ssl_transform_v2
    else:
        raise ValueError(f"Unknown method_name: {method_name}")
    
    full_train_for_train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=selected_train_transform)
    full_train_for_val = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=test_transform)
    
    n_total = len(full_train_for_train)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    
    trainset = Subset(full_train_for_train, train_idx)
    valset = Subset(full_train_for_val, val_idx)
    testset = torchvision.datasets.CIFAR10(root=root, train=False, 
                                           download=True, transform=test_transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                              shuffle=True, num_workers=2)
    
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, 
                                              shuffle=False, num_workers=2)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                              shuffle=False, num_workers=2)
    
    return trainloader, valloader, testloader
