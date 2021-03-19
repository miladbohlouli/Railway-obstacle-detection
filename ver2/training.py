import numpy as np
from tqdm import tqdm
import torch
import sys
from helpers import iouCalc, visim, vislbl
import os
import cv2
from arguments import get_args
from utils import get_model, get_dataloader, get_transforms, draw_boxes, get_device, show_frame, show_video, save_video
from dataset import dataset, COCO_INSTANCE_CATEGORY_NAMES


def evaluate(model, dataset_loader, device, args):
    model.eval()
    resulted_frames = []
    with torch.no_grad():
        for i, (input_frame, original_frame) in tqdm(enumerate(dataset_loader)):
            out = model(input_frame)
            original_frame = original_frame.detach().numpy().squeeze(0)
            pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(out[0]['labels'].numpy())]
            pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(out[0]['boxes'].detach().numpy())]
            pred_score = out[0]['scores'].detach().numpy()

            if len(pred_boxes) > 0 and (pred_score > args.threshold).any():
                pred_t = np.where(pred_score > args.threshold)[0][-1]
                pred_boxes = pred_boxes[:pred_t+1]
                pred_class = pred_class[:pred_t+1]

                # Draw the rectangles
                original_frame = draw_boxes(original_frame, pred_boxes, pred_class, args)
            resulted_frames.append(original_frame)
            # show_frame(original_frame, str(i))

    return resulted_frames


if __name__ == '__main__':
    args = get_args()
    model = get_model(args)
    transforms = get_transforms(args)
    ds = dataset(transforms, args)
    data_loader = get_dataloader(ds, args)
    device = get_device()
    edited_frames = evaluate(model, data_loader, device, args)
    print("showing the video")
    save_video(edited_frames, fps=25)
    show_video(edited_frames)

