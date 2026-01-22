import torch
import argparse


from datasets import get_supervised_dataloaders, get_rotnet_dataloaders, get_simclr_dataloaders
from models import get_encoder
from methods import get_method
from utils import set_seed, evaluate, evaluate_rotnet


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--model", default="resnet20")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--method", default="supervised_learning")
    parser.add_argument("--seed", type=int, default=42)
    
    # SimCLR Options(added)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--proj_hidden_dim", type=int, default=512)
    parser.add_argument("--proj_output_dim", type=int, default=128)
    
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # encoder    
    encoder = get_encoder(args.model)

    # dataset + method
    Method = get_method(args.method)
    
    if args.method == "supervised_learning":
        num_classes = 10
        trainloader, testloader = get_supervised_dataloaders(batch_size=args.batch_size)
        model = Method(encoder, num_classes).to(device)
    
    elif args.method == "rotnet":
        trainloader, testloader = get_rotnet_dataloaders(batch_size=args.batch_size)
        model = Method(encoder).to(device)
        
    elif args.method == "simclr":
        trainloader, testloader = get_simclr_dataloaders(batch_size=args.batch_size)
        model = Method(encoder, temperature=args.temperature, proj_hidden_dim=args.proj_hidden_dim,
                       proj_output_dim=args.proj_output_dim).to(device)
        

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        # for supervised learning/rotnet
        if args.method in ['supervised_learning', 'rotnet']:
            for x, y in trainloader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                loss = model((x, y))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        elif args.method == 'simclr':
            for (x1, x2), _ in trainloader:
                x1 = x1.to(device)
                x2 = x2.to(device)
                
                optimizer.zero_grad()
                loss = model((x1, x2))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
        
        avg_loss = running_loss / len(trainloader)
        print(f"epoch {epoch+1}/{args.epochs} | train loss {avg_loss:.4f}")

    if testloader is not None:
        if args.method == "supervised_learning":
            test_acc = evaluate(model, testloader, device)
        elif args.method == "rotnet":
            test_acc = evaluate_rotnet(model, testloader, device)
        else:
            test_acc = None
    
        if test_acc is not None:
            print(f"Final Test Accuracy: {test_acc*100:.2f}%")
    
if __name__ == "__main__":
    main()
