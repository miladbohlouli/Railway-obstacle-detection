from arguments import get_args
import logging
from utils import get_model, get_video_input_output, show_frame, get_video_writer
from processing import process_input
import cv2 as cv

logging.basicConfig(level=logging.INFO)


def main():
    args = get_args()
    model = get_model(args)
    cap, out_file = get_video_input_output(args)
    video_writer = get_video_writer(cap, out_file)

    # process the input
    process_input(model, cap, video_writer, out_file, args)


if __name__ == '__main__':
    main()
