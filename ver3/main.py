from arguments import get_args
import logging
from utils import get_model, get_video_input_output, get_video_writer, read_labels, cal_metrics
from processing import process_input
from sklearn.metrics import f1_score, accuracy_score

logging.basicConfig(level=logging.INFO)


def main():
    args = get_args()
    model = get_model(args)
    cap, out_file = get_video_input_output(args)
    video_writer = get_video_writer(cap, out_file)

    # process the input
    predicted_states = process_input(model, cap, video_writer, out_file, args)
    labels = read_labels(args)

    # calculate the quantitative measures
    cal_metrics(predicted_states, labels)


if __name__ == '__main__':
    main()
