from utils import get_dataloader, get_model, get_device
from arguments import get_args
from training import *
import torch
from torch import optim
from torch import nn
from matplotlib import pyplot as plt

def main():
    args = get_args()
    model = get_model(args)
    data_loader = get_dataloader(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.lr_momentum, weight_decay=args.lr_weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Specify the device
    device = get_device()
    print(f"____Running the model on {device}___\n")

    print("____Training the model from scratch____") if args.train else print("___Fine tuning the model____")


if __name__ == '__main__':
    main()
