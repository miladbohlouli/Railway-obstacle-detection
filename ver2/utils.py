import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models
import os
import cv2
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
import copy


def get_dataloader(dataset, args):
    return DataLoader(dataset,
                      batch_size=1,
                      shuffle=False,
                      num_workers=args.num_workers)


def get_model(args):
    if args.model == "fastercnn_resnet50_fpn":
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    else:
        raise Exception("The specified model is not correct!!")

    return model


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def read_video(path="videoplayback.mp4", show=False):
    cap = cv2.VideoCapture(path)
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    cap.release()

    if show:
        show_video(frames, "original movie")
    return frames


def show_video(frames: list, name="frame"):
    for i, frame in enumerate(frames):
        cv2.imshow(name, frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def save_video(frames, args, fps=25):
    frame_width, frame_height = frames[0].shape[1], frames[0].shape[0]
    out = cv2.VideoWriter(os.path.join(args.save_path, "saved_video.avi"), cv2.VideoWriter_fourcc("M", "J", "P", "G"), fps, (frame_width, frame_height))
    for frame in tqdm(frames):
        out.write(frame)
    out.release()


def get_transforms(args):
    return transforms.Compose([
        # transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(args.dataset_mean,
                             args.dataset_std)
    ])


def show_frame(frame, name="test", save=False, args=None):
    plt.imshow(frame)
    if save:
        cv2.imwrite(os.path.join(args.save_path, name), frame)
    plt.show()



def convert_np_PIL(np_array: np.ndarray):
    """
    Read the np array and convert it to float and scale it between 0 and 1 with data type of float32
    :param np_array: The input image
    :return: Image with format PIL.Image in range [0, 1]
    """
    return Image.fromarray(np_array)


def draw_boxes(frame, boxes, pred_cls, args):
    edited_frame = copy.deepcopy(frame)
    if len(boxes) == 0:
        print("There are no objects detected!!")
    for i, (box, clas) in enumerate(zip(boxes, pred_cls)):
        cv2.rectangle(edited_frame, box[0], box[1], args.color, args.rectangle_thickness)
        cv2.putText(edited_frame, clas, box[0], cv2.FONT_HERSHEY_SIMPLEX, args.text_size, args.color, args.text_thickness)
    return edited_frame

