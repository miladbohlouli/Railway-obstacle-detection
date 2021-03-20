import argparse

conf_url = "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true"
weights_url = "https://pjreddie.com/media/files/yolov3.weights"
class_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
               'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def get_args():
    parser = argparse.ArgumentParser(description='VIPriors Segmentation baseline training script')

    # model parameters
    parser.add_argument('--model_conf', default='model/yolov3.cfg',
                        type=str, help='Path to the model configuration')

    parser.add_argument('--model_weights', default='model/yolov3.weights',
                        type=str, help='Path to the model weights')

    parser.add_argument('--roi_vicinity', default=5, type=int,
                        help="The percentage of the vicinity that will be added to the RoI")

    parser.add_argument('--conf_threshold', default=0.5, type=float,
                        help="The threshold used for detection of the objects, the higher the "
                             "value of this parameter, the more uncertain objects will be omitted ")

    parser.add_argument('--nms_threshold', default=0.4, type=float,
                        help="The threshold used for non-maximum suppression")

    # source and destination directories
    parser.add_argument('--video', metavar='path/to/source/video', type=str,
                        help='path to the video source file')
    parser.add_argument('--image', metavar='path/to/source/image', type=str,
                        help='path to the image input')

    parser.add_argument('--save_path', metavar='path/to/save_results', default='',
                        type=str, help='path to results saved')

    # data augmentation hyper-parameters
    parser.add_argument('--image_size', default=[416, 416], nargs='+', type=int, help='image size during training')
    parser.add_argument('--threshold', type=float, default=0.9, help="The threshold the boxes will be chosen with")

    # appearance parameters
    parser.add_argument('--text_size', default=3, help='Text size used for writing on the images')
    parser.add_argument('--text_thickness', default=3, help='Thickness of the written text')
    parser.add_argument('--rectangle_thickness', default=3, help='Thickness of the rectangle border')
    parser.add_argument('--color', default=(0, 255, 0), help='color for drawing on the image')

    args = parser.parse_args()
    return args