from torch.utils.data import Dataset
from utils import read_video, convert_np_PIL
import os
import numpy as np
from utils import show_video

COCO_INSTANCE_CATEGORY_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'car', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

class dataset(Dataset):
    def __init__(self, transforms, args):
        if os.path.isdir(args.source_path):
            files_list = np.array(os.listdir(args.source_path))
            files_checked = np.array([(".mp4" or ".mkv") in file_name for file_name in files_list])
            if files_checked.all() is False:
                raise Exception("There is no files detected with the supported formats: .mkv, .mp4")
            else:
                source_file_path = files_list[files_checked][0]

        elif os.path.isfile(args.source_path):
            source_file_path = args.source_path

        else:
            raise Exception("The specified path or file for source_video is not valid!!")

        self.transforms = transforms
        self.frames = read_video(source_file_path, show=False)[200:250]

    def __getitem__(self, idx):
        return self.transforms(convert_np_PIL(self.frames[idx])), self.frames[idx]

    def __len__(self):
        return self.frames.__len__()


if __name__ == '__main__':
    pass

