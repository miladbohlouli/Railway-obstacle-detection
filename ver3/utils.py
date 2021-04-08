import os
import cv2 as cv
import requests
import logging
import sys
from arguments import download_urls, class_names
import numpy as np
import joblib
import imutils


STATUS = {
    "DANGER": (0, 0, 255),
    "WARNING": (255, 0, 255),
    "SAFE": (0, 255, 0)
}


def get_model(args):
    logging.debug("___Initiating the model___")

    if args.model_path is None or not os.path.isfile(args.model_path):
        if not os.path.isdir("model"):
            os.mkdir("model")

        model_configuration = os.path.join("model", args.model + ".cfg")
        model_weights = os.path.join("model", args.model + ".weights")

    else:
        model_configuration = os.path.join(args.model_path, args.model + ".cfg")
        model_weights = os.path.join(args.model_path, args.model + ".weights")

    if not os.path.isfile(model_configuration):
        path = os.path.join("model", args.model + ".cfg")
        logging.info(f"___Downloading the configuration of the network in {path}___")
        r = requests.get(download_urls[args.model+"_conf"], allow_redirects=True)
        open(path, "wb").write(r.content)
        model_configuration = path

    if not os.path.isfile(model_weights):
        path = os.path.join("model", args.model + ".weights")
        logging.info(f"___Downloading the weights of the network in {path}___")
        r = requests.get(download_urls[args.model+"_weights"], allow_redirects=True)
        open(path, "wb").write(r.content)
        model_weights = path

    logging.info(f"___Loading the model configuration from {model_configuration}___")
    logging.info(f"___Loading the model weights from {model_weights}___")
    return cv.dnn.readNetFromDarknet(model_configuration, model_weights)


def get_video_input_output(args):
    logging.debug("___Constructing the video reader___")
    if args.video:
        if not os.path.isfile(args.video):
            logging.error(f"___The specified file for video from {args.video} does not exist___")
            sys.exit(1)
        logging.info(f"___Loading the source video in {args.video}___")
        cap = cv.VideoCapture(args.video)
        output_file = args.video[:-4].split("/")[-1] + "_yolo_out.avi"
    elif args.image:
        if not os.path.isfile(args.image):
            logging.error(f"___The specified file for image in {args.image} does not exist___")
            sys.exit(1)
        logging.info(f"___Loading the source image in {args.image}___")
        cap = cv.VideoCapture(args.image)
        output_file = args.image[:-4].split("/")[-1] + "_yolo_out.jpg"
    else:
        # Source from the webcam
        logging.error(f"___The specified source files are not valid___")
    logging.debug("___The video reader opened successfully___")

    if args.save_path and os.path.isdir(args.save_path):
        output_file = os.path.join(args.save_path, output_file)
    else:
        if args.save_path:
            logging.info(f"___The specified saving path ({args.save_path}) is not valid___")
            logging.info(f"___Saving in the default path instead: saved_results/")

        if not os.path.isdir("saved_results"):
            os.mkdir("saved_results")
        output_file = os.path.join("saved_results", output_file)

    return cap, output_file


def get_video_writer(cap, output_file):
    return cv.VideoWriter(output_file, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
            (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))


def select_region_of_interest(frame, args):
    logging.info("___Please select the region of interest___")
    roi = cv.selectROI(frame, showCrosshair=False)
    cv.destroyAllWindows()

    # Add margin according to the specified value in args.RoI_vicinity
    r = [int(roi[1] - roi[3] * args.roi_vicinity/100), int(roi[0] - roi[2] * args.roi_vicinity/100),
         int(roi[3] + (2 * args.roi_vicinity / 100 * roi[3])), int(roi[2] + (2 * args.roi_vicinity / 100 * roi[2]))]
    logging.debug(f"___The selected region {r}___")
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


def draw_pred(frame, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if class_names:
        assert (classId < len(class_names))
        label = '%s:%s' % (class_names[classId], label)

    logging.debug(f"___Drawing the prediction for {label}___")
    # Display the label at the top of the bounding box
    label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, label_size[1])
    cv.rectangle(frame, (left, top - round(1.5 * label_size[1])), (left + round(1.5 * label_size[0]), top + base_line),
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


def get_region_of_interest(image):
    global pts
    global img
    pts = []
    img = image.copy()
    cv.namedWindow('Select region of interest')
    cv.setMouseCallback('Select region of interest', draw_roi)
    cv.imshow('Select region of interest', img)

    while True:
        key = cv.waitKey(1) & 0xFF
        if key == ord("c"):
            cv.destroyAllWindows()
            return None

        if key == 27:
            sys.exit()

        if key == 13:
            cv.destroyAllWindows()
            return pts


def draw_roi(event, x, y, flags, param):
    img2 = img.copy()

    if event == cv.EVENT_LBUTTONDOWN and flags != 17:   # Left click, select point
        pts.append((x, y))

    if event == cv.EVENT_LBUTTONDOWN and flags == 17:   # Shift + Left click --> remove the last selected point
        pts.pop()

    if len(pts) > 0:
        # Draw the last point in pts
        cv.circle(img2, pts[-1], 3, (0, 0, 255), -1)

    if len(pts) > 1:
        for i in range(len(pts) - 1):
            cv.circle(img2, pts[i], 5, (0, 0, 255), -1)  # x ,y is the coordinates of the mouse click place
            cv.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)

    cv.imshow('Select region of interest', img2)


def draw_polylines(image, points):
    points = np.array(points, np.int32)
    points = points.reshape((-1, 1, 2))
    image = cv.polylines(image, [points], True, (0, 0, 255), 2)
    return image


def infer_status(frame, object, roi, args):

    points = np.array(roi, np.int32)
    points = points.reshape((-1, 1, 2))

    mask1 = np.zeros((frame.shape[0], frame.shape[1])).astype(np.int8)
    # mask1 = np.zeros((frame.shape[0], frame.shape[1]))
    mask1[object[1]:object[3], object[0]:object[2]] = 1

    mask2 = np.zeros((frame.shape[0], frame.shape[1]))
    mask2 = cv.fillPoly(mask2.copy(), [points], 1).astype(np.int8)

    intersection = np.logical_and(mask1, mask2)
    intersection_rate = np.sum(intersection) / np.sum(mask1)

    if intersection_rate > args.danger_threshold:
        status = "DANGER"
    elif 0 < intersection_rate < args.danger_threshold:
        status = "WARNING"
    else:
        status = "SAFE"

    return status
