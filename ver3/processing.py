import cv2 as cv
import logging
import os
from utils import get_output_names, draw_preds, get_region_of_interest
import numpy as np

# These are the available states for the system to be in


def process_input(model, cap, video_writer, output_file, args):

    # Selecting the region of interest
    # roi = get_region_of_interest(cap)
    # print(roi)

    while cv.waitKey(1) < 0:
        has_frame, frame = cap.read()

        frame = pre_process(frame)

        # Check if the input is finished
        if not has_frame:
            logging.info("___Done processing the input___")
            logging.info(f"___Saving the file in {output_file}___")
            cv.waitKey(3000)
            cap.release()
            break

        blob = cv.dnn.blobFromImage(frame, 1 / 255, (args.image_size[0], args.image_size[1]), [0, 0, 0], 1, crop=False)

        model.setInput(blob)

        outs = model.forward(get_output_names(model))



        post_process(frame, outs, args)

        # Todo: Use this parameter to make th model real-time
        t, _ = model.getPerfProfile()

        if args.image:
            cv.imwrite(output_file, frame.astype(np.uint8))
        else:
            video_writer.write(frame.astype(np.uint8))

        cv.imshow("test", frame)


def post_process(frame, outs, args):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    logging.debug("___Filtering the detected objects___")
    outs = np.concatenate(outs, axis=0)
    maximum_confs = np.max(outs[:, 5:], axis=1)
    object_index = np.where(maximum_confs > args.conf_threshold)[0]
    detected_objects = outs[object_index]
    center_x = (detected_objects[:, 0] * frame_width)
    center_y = (detected_objects[:, 1] * frame_height)
    width = (detected_objects[:, 2] * frame_width)
    height = (detected_objects[:, 3] * frame_height)
    left = (center_x - width / 2)
    top = (center_y - height / 2)
    boxes = [[int(left[i]), int(top[i]), int(width[i]), int(height[i])] for i in range(len(detected_objects))]
    confidences = list(maximum_confs[object_index].astype(np.float))
    class_ids = np.argmax(outs[object_index, 5:], axis=1)

    logging.debug("___Applying the non-maximum suppression___")
    indices = cv.dnn.NMSBoxes(boxes, confidences, args.conf_threshold, args.nms_threshold)

    logging.debug("___Drawing the detected objects___")
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        draw_preds(frame, class_ids[i], confidences[i], left, top, left + width, top + height)


def pre_process(frame):
    # Todo: add any pre-processing in this function
    return frame



