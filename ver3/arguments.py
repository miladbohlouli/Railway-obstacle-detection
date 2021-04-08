import argparse

download_urls = {
    "yolov3_conf": "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true",
    "yolov3_weights": "https://pjreddie.com/media/files/yolov3.weights",
    "yolov3-tiny_conf": "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg?raw=true",
    "yolov3-tiny_weights": "https://pjreddie.com/media/files/yolov3-tiny.weights",
    "yolov4_conf": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg?raw=true",
    "yolov4_weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights",
    "yolov4-tiny_conf": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg?raw=true",
    "yolov4-tiny_weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
}

original_class_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
               'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

class_names = ['person', 'vehicle', 'vehicle', 'bike', 'vehicle', 'vehicle', 'train', 'vehicle', 'boat', 'ignore',
               'ignore', 'ignore', 'ignore', 'ignore', 'animal', 'animal', 'animal', 'animal', 'animal', 'animal',
               'animal', 'animal', 'animal', 'animal', 'other', 'other', 'other', 'ignore', 'other', 'other',
               'ignore', 'other', 'other', 'other', 'other', 'other', 'other', 'other',
               'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other',
               'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other',
               'ignore', 'ignore', 'ignore', 'ignore', 'ignore', 'other', 'other', 'other', 'other',
               'other', 'ignore', 'ignore', 'ignore', 'ignore', 'ignore', 'other', 'ignore', 'ignore',
               'other', 'other', 'other', 'other']


def get_args():
    parser = argparse.ArgumentParser(description='VIPriors Segmentation baseline training script')

    # model parameters

    parser.add_argument('--model', default="yolov4-tiny", type=str,
                        help="The type of the model which will be used")

    # source and destination directories
    parser.add_argument('--model_path', type=str, help='Path to the model directory')

    parser.add_argument('--video', metavar='path/to/source/video', type=str, help='path to the video source file')

    parser.add_argument('--image', metavar='path/to/source/image', type=str, help='path to the image input')

    parser.add_argument('--save_path', metavar='path/to/save_results', type=str, help='path to saving directory')

    # data augmentation hyper-parameters
    parser.add_argument('--image_size', default=[416, 416], nargs='+', type=int, help='image size during training')
    parser.add_argument('--danger_threshold', default=0.1, help="This is the threshold which defines the danger zone")
    parser.add_argument('--inertia_threshold', default=20,
                        help="This is the inertia for danger and warning state lingering")

    parser.add_argument('--conf_threshold', default=0.7, type=float,
                        help="The threshold used for detection of the objects, the higher the "
                             "value of this parameter, the more uncertain objects will be omitted ")

    parser.add_argument('--nms_threshold', default=0.4, type=float,
                        help="The threshold used for non-maximum suppression")

    # appearance parameters
    parser.add_argument('--text_size', default=3, help='Text size used for writing on the images')
    parser.add_argument('--text_thickness', default=3, help='Thickness of the written text')
    parser.add_argument('--rectangle_thickness', default=3, help='Thickness of the rectangle border')
    parser.add_argument('--color', default=(0, 255, 0), help='color for drawing on the image')

    # Other parameters
    parser.add_argument('--req_fps', default=60, help='required fps')

    args = parser.parse_args()
    return args