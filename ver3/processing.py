import cv2 as cv
import logging
from utils import get_output_names, draw_pred, get_region_of_interest, draw_polylines, infer_status, STATUS
import numpy as np
import time
from arguments import class_names


def process_input(model, cap, video_writer, output_file, args):

    # Selecting the region of interest
    _, first_frame = cap.read()

    logging.info("___Selecting the region of Interest___")
    print("============================Controls for Region selection============================\n"
          "+    Left mouse key:         add point\n"
          "+    Shift + Left mouse key: remove the last point\n"
          "+    Enter:                  Finalize the selected points\n"
          "+    c:                      Clear the selections\n"
          "======================================================================================\n")

    roi_points = None
    while roi_points is None:
        roi_points = get_region_of_interest(first_frame)

    flag = True
    inertia_counter = 0
    skip_frames = 0
    status = "WARNING"
    logging.debug(f"___The selected points include {roi_points}___")
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

        if skip_frames == 0:

            # Get output of the model
            start_time = time.time()
            blob = cv.dnn.blobFromImage(frame, 1 / 255, (args.image_size[0], args.image_size[1]), [0, 0, 0], 1, crop=False)
            model.setInput(blob)
            outs = model.forward(get_output_names(model))

            # This is the reference time which will be used to make the model real time by calculating the skip frames
            #      between among the frames to make the model function in a descent way
            reference_time = time.time() - start_time
            skip_frames = int((args.req_fps * reference_time)) + 1
            # One has been added to take into consideration the drawing and inferring process in to
            #   real time process of the system

        else:
            logging.debug("___Skipping the frame to make the results real time___")
            skip_frames -= 1

        # Infer the status regarding the roi and the detected objects
        momentary_status = post_process(frame, outs, roi_points, args)

        # frame = draw_polylines(frame, roi_points)

        # Add the inertia technique to make the model more prone to noise
        if momentary_status == "DANGER":
            inertia_counter = 0
            flag = True
            status = momentary_status

        elif momentary_status == "SAFE" and flag:
            inertia_counter += 1

        if inertia_counter == args.inertia_threshold:
            status = "SAFE"
            inertia_counter = 0
            flag = False

        cv.putText(frame, "system status: " + status, (0, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, STATUS[status], 2,
                   cv.LINE_AA)
        cv.imshow("real time result", frame)

        if args.image:
            cv.imwrite(output_file, frame.astype(np.uint8))
        else:
            video_writer.write(frame.astype(np.uint8))


def post_process(frame, outs, roi, args):
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

    status = "SAFE"
    logging.debug("___Checking the status of the region of interest___")
    for i in indices:

        i = i[0]
        box = boxes[i]
        left, top, width, height = box[0], box[1], box[2], box[3]

        if not class_names[class_ids[i]] == "ignore":
            draw_pred(frame, class_ids[i], confidences[i], left, top, left + width, top + height)

            status = infer_status(frame, [left, top, left + width, top + height], roi, args)

    return status


def pre_process(frame):
    # Todo: add any pre-processing in this function
    return frame



