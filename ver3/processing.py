import cv2 as cv
import logging
import os
from utils import get_output_names, draw_preds
import numpy as np

def process_input(model, cap, video_writer, output_file, args):

    path = os.path.join(args.save_path, output_file)
    while cv.waitKey(1) < 0:
        has_frame, frame = cap.read()

        frame = pre_process(frame)

        # Check if the input is finished
        if not has_frame:
            logging.info("___Done processing the input___")
            logging.info(f"___Saving the file in {path}___")
            cv.waitKey(3000)
            cap.release()
            break

        blob = cv.dnn.blobFromImage(frame, 1 / 255, (args.image_size[0], args.image_size[1]), [0, 0, 0], 1, crop=False)

        model.setInput(blob)

        outs = model.forward(get_output_names(model))

        post_process(frame, outs, args)

        #Todo: Use this parameter to make th model real-time
        t, _ = model.getPerfProfile()

        cv.imshow("test", frame)



def post_process(frame, outs, args):
    # Todo: Add any postprocessing to the detected objects in this section
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > args.conf_threshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, args.conf_threshold, args.nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        draw_preds(frame, classIds[i], confidences[i], left, top, left + width, top + height)


def pre_process(frame):
    # Todo: add any pre-processing in this function
    return frame



