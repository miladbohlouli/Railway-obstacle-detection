import os
import cv2 as cv
import requests
import logging
import sys
from arguments import download_urls, class_names

def get_model(args):
    logging.debug("___Initiating the model___")
    model_configuration = os.path.join(args.model_path, args.model + ".cfg")
    model_weights = os.path.join(args.model_path, args.model + ".weights")

    if not os.path.isfile(model_configuration):
        if not os.path.isdir("model"):
            os.mkdir("model")
        path = os.path.join("model", args.model + ".cfg")
        logging.info(f"___Downloading the configuration of the network in {path}___")
        r = requests.get(download_urls[args.model], allow_redirects=True)
        open(path, "wb").write(r.content)
        model_configuration = path

    if not os.path.isfile(model_weights):
        if not os.path.isdir("model"):
            os.mkdir("model")
        path = os.path.join("model", args.model + ".weights")
        logging.info(f"___Downloading the weights of the network in {path}___")
        r = requests.get(download_urls[args.model], allow_redirects=True)
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
            os.mkdir("save_results")
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


def draw_preds(frame, classId, conf, left, top, right, bottom):
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
