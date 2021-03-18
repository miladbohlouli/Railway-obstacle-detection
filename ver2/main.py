from minicity import MiniCity
from utils import get_dataloader, get_model, get_device, save_model, load_model
from arguments import get_args
from training import *
import torch
from torch import optim
from torch import nn
from matplotlib import pyplot as plt

def main():
    args = get_args()
    model = get_model(MiniCity, args)
    data_loader = get_dataloader(MiniCity, args)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.lr_momentum, weight_decay=args.lr_weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=MiniCity.voidClass)
    start_epoch = 0
    train_losses = []
    eval_losses = []
    mean_ious = []
    nz_ious = []

    # Specify the device
    device = get_device()
    print(f"____Running the model on {device}___\n")

    print("____Training the model from scratch____") if args.train else print("___Fine tuning the model____")

    if args.load:
        saving_dict = load_model(args.save_path)
        model.load_state_dict(saving_dict["model"])
        optimizer.load_state_dict(saving_dict["optimizer"])
        start_epoch = saving_dict["epoch"] + 1
        scheduler = saving_dict["scheduler"]
        train_losses = list(saving_dict["train_losses"])
        eval_losses = list(saving_dict["eval_losses"])
        mean_ious = list(saving_dict["mean_ious"])
        nz_ious = list(saving_dict["mean_nz_ious"])

    for epoch in range(start_epoch, start_epoch + args.epochs):

        # training step
        train_loss = train_epoch(model, data_loader["train"], optimizer, criterion, scheduler, device, args)

        # validation step
        mean_iou, nz_iou, eval_loss = eval_epoch(model, data_loader["val"], criterion, MiniCity.classLabels, MiniCity.validClasses,
                             MiniCity.mask_colors, epoch, args.save_path, device, args)

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        mean_ious.append(mean_iou)
        nz_ious.append(nz_iou)

        print(f"({epoch + 1}/{start_epoch + args.epochs}) ---> \ttrain_loss:{train_loss:.2f}, \tval_loss: {eval_loss:.2f}")

    # Saving the results
    plt.plot(range(epoch+1), train_losses)
    plt.plot(range(epoch+1), eval_losses)
    plt.legend(["training loss", "validation loss"])
    plt.savefig(f"saved_results/loss_values_{epoch + 1}.png")

    plt.figure()
    plt.plot(range(epoch + 1), mean_ious)
    plt.plot(range(epoch + 1), nz_ious)
    plt.title("Mean IoU on the validation results")
    plt.legend(["All classes", "common classes"])
    plt.savefig(f"saved_results/miou_values_{epoch + 1}.png")

    saving_dict = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler,
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "mean_ious": mean_ious
    }

    save_model(args.save_path, saving_dict)

if __name__ == '__main__':
    main()
