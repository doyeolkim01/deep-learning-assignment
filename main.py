import torch
import argparse

from datasets import get_dataloaders
from models import get_encoder
from methods import get_method
from utils import set_seed, to_device, evaluate, evaluate_rotnet, knn_evaluation


def parse_args():
    # parse command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--model", default="resnet20")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--method", default="supervised_learning")
    parser.add_argument("--seed", type=int, default=42)
    
    # SimCLR Options
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--proj_hidden_dim", type=int, default=512)
    parser.add_argument("--proj_output_dim", type=int, default=128)
    
    # MoCo Options
    parser.add_argument("--queue_size", type=int, default=4096)
    parser.add_argument("--momentum", type=float, default=0.999)
    
    # SimSiam options
    parser.add_argument("--pred_hidden_dim", type=int, default=512)
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed) # set random seed
    
    if args.dataset.lower() != "cifar10":
        raise ValueError("Only cifar10 is supported now!")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # build encoder network    
    encoder = get_encoder(args.model)

    # get method for training
    Method = get_method(args.method)
    
    # load dataset
    trainloader, feature_bank_loader, valloader, testloader = get_dataloaders(args.method, args.batch_size, img_size = 32, val_ratio = 0.2)
    
    # select model
    if args.method == "supervised_learning":
        num_classes = 10
        model = Method(encoder, num_classes).to(device)
    
    elif args.method == "rotnet":
        model = Method(encoder).to(device)
        
    elif args.method == "simclr":
        model = Method(encoder, temperature=args.temperature, proj_hidden_dim=args.proj_hidden_dim,
                       proj_output_dim=args.proj_output_dim).to(device)
        
    elif args.method in ["moco_v1", "moco_v2"]:
        if args.method == 'moco_v1':
            version = "v1" 
        else:
            version = "v2"

        model = Method(encoder, version=version, temperature=args.temperature, proj_hidden_dim=args.proj_hidden_dim, 
                       proj_output_dim=args.proj_output_dim, queue_size=args.queue_size, momentum=args.momentum).to(device)
            
    elif args.method == "byol":
        model = Method(encoder, hidden_dim=args.proj_hidden_dim, output_dim=args.proj_output_dim, momentum = 0.99).to(device)
        
    elif args.method == "simsiam":
        model = Method(encoder, proj_hidden_dim=args.proj_hidden_dim, proj_output_dim=args.proj_output_dim, 
                       pred_hidden_dim=args.pred_hidden_dim).to(device)
     
    else:
        raise ValueError(f"Unknown method: {args.method}")
        

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch in trainloader:
            batch = to_device(batch, device)
                
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            
            if args.method in ['moco_v1', 'moco_v2', 'byol']:
                model.momentum_update() 

            running_loss += loss.item()
        
        avg_loss = running_loss / len(trainloader)
        print(f"epoch {epoch+1}/{args.epochs} | train loss {avg_loss:.4f}")

    # evaluation
    if args.method == "supervised_learning":
        test_acc = evaluate(model, testloader, device)
        print(f"Final Test Accuracy: {test_acc*100:.2f}%")
        
    elif args.method == "rotnet":
        test_acc = evaluate_rotnet(model, testloader, device)
        print(f"Final Rotation Test Accuracy: {test_acc*100:.2f}%")
        
    elif args.method in ["simclr", "moco_v1", "moco_v2", "byol", "simsiam"]:
        val_acc = knn_evaluation(model, feature_bank_loader, valloader, device)
        print(f"Final kNN Val Accuracy(k=1): {val_acc*100:.2f}%")
        
        
if __name__ == "__main__":
    main()
