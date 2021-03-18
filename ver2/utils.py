import torchvision.transforms.functional as TF
import numpy as np
import torch
from PIL import ImageOps
from torchvision import models
from models import *
import os

def get_dataloader(dataset, args):
    # args = args

    def test_trans(image, mask=None):
        # Resize, 1 for Image.LANCZOS
        image = TF.resize(image, args.test_size, interpolation=1)
        # From PIL to Tensor
        image = TF.to_tensor(image)
        # Normalize
        image = TF.normalize(image, args.dataset_mean, args.dataset_std)

        if mask:
            # Resize, 0 for Image.NEAREST
            mask = TF.resize(mask, args.test_size, interpolation=0)
            mask = np.array(mask, np.uint8)  # PIL Image to numpy array
            mask = torch.from_numpy(mask)  # Numpy array to tensor
            return image, mask
        else:
            return image

    def train_trans(image, mask):
        # Generate random parameters for augmentation
        bf = np.random.uniform(1 - args.colorjitter_factor, 1 + args.colorjitter_factor)
        cf = np.random.uniform(1 - args.colorjitter_factor, 1 + args.colorjitter_factor)
        sf = np.random.uniform(1 - args.colorjitter_factor, 1 + args.colorjitter_factor)
        hf = np.random.uniform(-args.colorjitter_factor, +args.colorjitter_factor)
        pflip = np.random.randint(0, 1) > 0.5

        # Random scaling
        scale_factor = np.random.uniform(0.75, 2.0)
        scaled_train_size = [int(element * scale_factor) for element in args.train_size]

        # Resize, 1 for Image.LANCZOS
        image = TF.resize(image, scaled_train_size, interpolation=1)
        # Resize, 0 for Image.NEAREST
        mask = TF.resize(mask, scaled_train_size, interpolation=0)

        # Random cropping
        if not args.train_size == args.crop_size:
            if image.size[1] <= args.crop_size[0]:  # PIL image: (width, height) vs. args.size: (height, width)
                pad_h = args.crop_size[0] - image.size[1] + 1
                pad_w = args.crop_size[1] - image.size[0] + 1
                image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
                mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=19)

            # From PIL to Tensor
            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)
            h, w = image.size()[1], image.size()[2]  # scaled_train_size #args.train_size
            th, tw = args.crop_size

            i = np.random.randint(0, h - th)
            j = np.random.randint(0, w - tw)
            image_crop = image[:, i:i + th, j:j + tw]
            mask_crop = mask[:, i:i + th, j:j + tw]

            image = TF.to_pil_image(image_crop)
            mask = TF.to_pil_image(mask_crop[0, :, :])

        # H-flip
        if pflip == True and args.hflip == True:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Color jitter
        image = TF.adjust_brightness(image, bf)
        image = TF.adjust_contrast(image, cf)
        image = TF.adjust_saturation(image, sf)
        image = TF.adjust_hue(image, hf)

        # From PIL to Tensor
        image = TF.to_tensor(image)

        # Normalize
        image = TF.normalize(image, args.dataset_mean, args.dataset_std)

        # Convert ids to train_ids
        mask = np.array(mask, np.uint8)  # PIL Image to numpy array
        mask = torch.from_numpy(mask)  # Numpy array to tensor

        return image, mask

    trainset = dataset(args.dataset_path, split='train', transforms=train_trans)
    valset = dataset(args.dataset_path, split='val', transforms=test_trans)
    testset = dataset(args.dataset_path, split='test', transforms=test_trans)
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(trainset,
                                                       batch_size=args.batch_size, shuffle=True,
                                                       pin_memory=args.pin_memory, num_workers=args.num_workers)
    dataloaders['val'] = torch.utils.data.DataLoader(valset,
                                                     batch_size=args.batch_size, shuffle=False,
                                                     pin_memory=args.pin_memory, num_workers=args.num_workers)
    dataloaders['test'] = torch.utils.data.DataLoader(testset,
                                                      batch_size=args.batch_size, shuffle=False,
                                                      pin_memory=args.pin_memory, num_workers=args.num_workers)

    return dataloaders


def get_model(Dataset, args):
    pretrained = not args.train
    if args.model == "DeepLabv3_resnet50":
        model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
    elif args.model == "DeepLabv3_resnet101":
        model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
    elif args.model == "fcnet":
        model = models.segmentation.fcn_resnet101(pretrained=pretrained)

    # if the weights are loaded, and the goal is to fine-tune the model, fix the require_grad parameters
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier sections of the main model
    model.classifier = DeepLabHead(2048, len(Dataset.validClasses))
    model.aux_classifier = FCNHead(1024, len(Dataset.validClasses))

    return model

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

def save_model(save_dir, save_dict):
    torch.save(save_dict, os.path.join(save_dir, "checkpoint.pt"))

def load_model(save_dir):
    return torch.load(os.path.join(save_dir, "checkpoint.pt"))