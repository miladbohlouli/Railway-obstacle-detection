import os
import cv2 as cv
import requests
import logging
import sys
from arguments import conf_url, class_names, weights_url

def get_model(args):
    logging.debug("___Initiating the model___")
    model_configuration = args.model_conf
    model_weights = args.model_weights

    if not os.path.isfile(args.model_conf):
        if not os.path.isdir("model"):
            os.mkdir("model")
        path = "model/yolov3.cfg"
        logging.info(f"___Downloading the configuration of the network in {path}___")
        r = requests.get(conf_url, allow_redirects=True)
        open(path, "wb").write(r.content)
        model_configuration = "model/yolov3.cfg"

    if not os.path.isfile(args.model_weights):
        if not os.path.isdir("model"):
            os.mkdir("model")
        path = "model/yolov3.weights"
        logging.info(f"___Downloading the weights of the network in {path}___")
        r = requests.get(weights_url, allow_redirects=True)
        open(path, "wb").write(r.content)
        model_weights = "model/yolov3.weights"

    logging.debug("___The model loaded successfully___")
    logging.info(f"___Loading the model configuration from {model_configuration}___")
    logging.info(f"___Loading the model weights from {model_weights}___")
    return cv.dnn.readNetFromDarknet(model_configuration, model_weights)


def get_video_input(args):
    logging.debug("___Constructing the video reader___")
    if args.video:
        if not os.path.isfile(args.video):
            logging.error(f"___The specified file for video from {args.video} does not exist___")
            sys.exit(1)
        logging.info(f"___Loading the source video in {args.video}___")
        cap = cv.VideoCapture(args.video)
        output_file = args.video[:-4] + "_yolo_out.avi"
    elif args.image:
        if not os.path.isfile(args.image):
            logging.error(f"___The specified file for image in {args.image} does not exist___")
            sys.exit(1)
        logging.info(f"___Loading the source image in {args.image}___")
        cap = cv.VideoCapture(args.image)
        output_file = args.image[:-4] + "_yolo_out.jpg"
    else:
        # Source from the webcam
        logging.info(f"___Loading the source video from webcam___")
        cap = cv.VideoCapture(0)
    logging.debug("___The video reader opened successfully___")
    return cap, output_file


def get_video_writer(cap, output_file):
    return cv.VideoWriter(output_file, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
            (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))


def select_region_of_interest(frame, args):
    # select the original RoI
    roi = cv.selectROI(frame, showCrosshair=False)
    cv.destroyAllWindows()

    # Add margin according to the specified value in args.RoI_vicinity
    r = [int(roi[1] - roi[3] * args.roi_vicinity/100), int(roi[0] - roi[2] * args.roi_vicinity/100),
         int(roi[3] + (2 * args.roi_vicinity / 100 * roi[3])), int(roi[2] + (2 * args.roi_vicinity / 100 * roi[2]))]
    return r


def show_frame(frame, name="test", save=False, args=None):
    cv.imshow(name, frame)
    cv.waitKey(0)

    if save:
        cv.imwrite(os.path.join(args.save_path, name + ".jpg"), frame)
    return


def get_output_names(net):
    layers_names = net.getLayerNames()
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def draw_preds(frame, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if class_names:
        assert (classId < len(class_names))
        label = '%s:%s' % (class_names[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
