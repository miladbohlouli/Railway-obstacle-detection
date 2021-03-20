import numpy as np
from tqdm import tqdm
import torch
from utils import draw_boxes
from dataset import COCO_INSTANCE_CATEGORY_NAMES


def evaluate(model, dataset_loader, device, args):
    model.to(device)
    model.eval()
    resulted_frames = []
    with torch.no_grad():
        for i, (input_frame, original_frame) in tqdm(enumerate(dataset_loader)):
            input_frame = input_frame.to(device)
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

    return resulted_frames
