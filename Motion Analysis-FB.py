import cv2
import numpy as np
from tqdm import tqdm
from time import time
from matplotlib import pyplot as plt


def show_frame(frame, name):
    cv2.imshow(name, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_video(frames: list, name="frame"):
    for i, frame in enumerate(frames):
        cv2.imshow(name, frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def read_video(path="videoplayback.mp4", show=False):
    cap = cv2.VideoCapture(path)
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    cap.release()

    if show:
        show_video(frames, "original movie")
    return frames


def save_video(frames, save_url="saved_video.avi", fps=25):
    frame_width, frame_height = frames[0].shape[1], frames[0].shape[0]
    out = cv2.VideoWriter(save_url, cv2.VideoWriter_fourcc("M", "J", "P", "G"), fps, (frame_width, frame_height))
    for frame in tqdm(frames):
        out.write(frame)
    out.release()


def preprocess_frame(frame, smooth=False, return_red=False):
    if return_red:
        frame = frame[:, :, 2]
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if smooth:
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
    return frame


fb_parameters = dict(
    flow=None,
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)

if __name__ == '__main__':
    first_frame, final_frame = 102, 2242
    frames = read_video()[first_frame:final_frame]

    old_gray = preprocess_frame(frames[0])
    hsv = np.zeros_like(frames[0])
    hsv[..., 1] = 255
    new_frames_scene = []

    # processing the video
    for i, frame in tqdm(enumerate(frames[1:])):
        new_gray = preprocess_frame(frame)
        flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, **fb_parameters)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame', rgb)
        new_frames_scene.append(rgb)

        if cv2.waitKey(30) & 0xff == ord("q"):
            break

        elif cv2.waitKey(30) & 0xff == ord("s"):
            plt.subplot(211)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.xticks([]), plt.yticks([]), plt.title("Original frame")
            plt.subplot(212)
            plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            plt.xticks([]), plt.yticks([]), plt.title("Heatmap")
            plt.savefig(f"results2/{time()}.png")

        old_gray = new_gray.copy()

    save_video(new_frames_scene, f"result2.avi")
