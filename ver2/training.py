import numpy as np
from tqdm import tqdm
import torch
import sys
from helpers import iouCalc, visim, vislbl
import os
import cv2


def train_epoch(model, data_loader, optimizer, criterion, scheduler, device, args):
    if args.crop_size is not None:
        res = args.crop_size[0] * args.crop_size[1]
    else:
        res = args.train_size[0] * args.train_size[1]

    # Set model in training mode
    model.train()

    # Move the model to gpu is available
    model.to(device)

    losses = []
    with torch.set_grad_enabled(True):
        # Iterate over data.
        for epoch_step, (inputs, labels, _) in enumerate(data_loader):

            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            outputs = outputs['out']
            preds = torch.argmax(outputs, 1)

            # cross-entropy loss
            loss = criterion(outputs, labels)

            # backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            bs = inputs.size(0)  # current batch size
            losses.append(loss.item())
            corrects = torch.sum(preds == labels.data)
            nvoid = int((labels == 19).sum())
            acc = corrects.double() / (bs * res - nvoid)  # correct/(batch_size*resolution-voids)

            # output training info
            scheduler.step(np.mean(losses))

    return np.mean(losses)


def eval_epoch(model, data_loader, criterion, class_labels, valid_labels, maskColors, epoch, folder, device, args):
    iou = iouCalc(class_labels, valid_labels, voidClass=19)
    losses = []

    # Set the model in evaluation mode
    model.eval()
    model.to(device)

    with torch.no_grad():
        for epoch_step, (input_images, target_labels, file_path) in enumerate(data_loader):

            # Forward path
            input_images = input_images.float().to(device)
            target_labels = target_labels.long().to(device)

            outs = model(input_images)["out"]
            predicted_labels = torch.argmax(outs, dim=1)

            loss = criterion(outs, target_labels)
            losses.append(loss.item())

            # Calculating the IoU metrics
            iou.evaluateBatch(predicted_labels, target_labels)

            if epoch_step == 0 and maskColors is not None:
                for i in range(input_images.size(0)):
                    filename = os.path.splitext(os.path.basename(file_path[i]))[0]
                    # Only save inputs and labels once

                    if epoch == 0:

                        img = visim(input_images[i, :, :, :], args)
                        label = vislbl(target_labels[i, :, :], maskColors)
                        if len(img.shape) == 3:
                            print(img[:, :, ::-1].shape, folder + '/images/{}.png'.format(filename))
                            cv2.imwrite(folder + '/images/{}.png'.format(filename), img[:, :, ::-1])
                        else:
                            cv2.imwrite(folder + '/images/{}.png'.format(filename), img)
                        cv2.imwrite(folder + '/images/{}_gt.png'.format(filename), label[:, :, ::-1])

                    # Save predictions
                    pred = vislbl(predicted_labels[i, :, :], maskColors)
                    cv2.imwrite(folder + '/images/{}_epoch_{}.png'.format(filename, epoch), pred[:, :, ::-1])

        mean_IoU, nz_IoU = iou.outputScores()
        return mean_IoU, nz_IoU, np.mean(losses)


