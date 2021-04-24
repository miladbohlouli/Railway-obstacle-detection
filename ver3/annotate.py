import numpy as np
from utils import get_video_input_output
from arguments import get_args
import logging
import sys
import os
import cv2 as cv

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    args = get_args()
    cap, _ = get_video_input_output(args)
    path = os.path.join(args.video.split("/")[0], args.video[:-4].split("/")[1] + "_labels.txt")
    output_file = open(path, "w")
    frame_num = 0

    print("==================================Controls for frames annotation==================================\n"
          "+    s:      Set the label as safe\n"
          "+    d:      Set the label as dangerous\n"
          "+    esc:    Close the window and save the annotated frames\n"
          "===================================================================================================\n")

    while cv.waitKey(1) < 0:
        has_frame, frame = cap.read()
        cv.putText(frame, f"frame: {frame_num} ", (0, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)

        if not has_frame:
            cv.waitKey(300)
            break

        cv.imshow("Classify the frame as safe or danger", frame)

        key = cv.waitKey()
        # When s is pressed
        if key == 115:
            output_file.write("safe\n")
            logging.info(f"___frame {frame_num} set to Safe___")

        # When d is pressed
        elif key == 100:
            output_file.write("danger\n")
            logging.info(f"___frame {frame_num} set to danger___")

        # close the file
        elif key == 27:
            logging.info(f"___closing the windows___")
            break

        frame_num += 1

    logging.info(f"___{frame_num} frames were labeled and saved to {path}___")
    output_file.close()
    cap.release()
    sys.exit()