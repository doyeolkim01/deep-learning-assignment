import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Supervised Learning
def get_supervised_dataloaders(batch_size, root="./data"):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=root, train=True, 
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                              shuffle=True, num_workers=2,
    )

    testset = torchvision.datasets.CIFAR10(root=root, train=False, 
                                           download=True, transform=test_transform,
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                             shuffle=False, num_workers=2,
    )

    return trainloader, testloader


# Rotnet
def get_rotnet_dataloaders(batch_size, root="./data"):
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  trainset = torchvision.datasets.CIFAR10(root=root, train=True, 
                                          download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                            shuffle=True, num_workers=2)

  testset = torchvision.datasets.CIFAR10(root=root, train=False, 
                                         download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                           shuffle=False, num_workers=2)

  return trainloader, testloader



# SimCLR
class SimCLRTransform:
    def __init__(self, img_size=32):
        self.base_transform = transforms.Compose([
            # Randomly crop region and resize to img_size
            transforms.RandomResizedCrop(img_size, scale = (0.2, 1.0)),
            # Apply horizontalflip
            transforms.RandomHorizontalFlip(),
            # Randomly change color properties(brightness,contrast,saturation,hue) with p=0.8
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p = 0.8),
            # Covert to grayscale with p=0.2
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
    
    def __call__(self, x):
        x1 = self.base_transform(x)
        x2 = self.base_transform(x)
        
        return (x1, x2)



def get_simclr_dataloaders(batch_size, root="./data"):
    simclr_transform = SimCLRTransform(img_size=32)
    
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, 
                                            download=True, transform=simclr_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                              shuffle=True, num_workers=2)

    # evaluation용 test loader
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    testset = torchvision.datasets.CIFAR10(root=root, train=False, 
                                           download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                             shuffle=False, num_workers=2)

    return trainloader, testloader


