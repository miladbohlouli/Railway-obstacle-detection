import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from time import time


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


def check_inside(rect:list, keypoint):
    if (rect[0] < keypoint[0] < rect[0] + rect[2]) and (rect[1] < keypoint[1] < rect[1] + rect[3]):
        return True
    return False

def check_keypoints(rect, keypoints):
    count = 0
    for keypoint in keypoints:
        if check_inside(rect, keypoint):
            count += 1
    return count

lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            # criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

feature_params = dict(
    maxCorners=30,
    qualityLevel=0.6,
    minDistance=10,
    blockSize=7
)
recalculate_every_n_frame = 100
max_keypoints_allowed = 3
min_displacement = 0.5

if __name__ == '__main__':
    frames = read_video("1.mp4")
    old_gray = preprocess_frame(frames[0])

    rect = cv2.selectROI(frames[0], showCrosshair=False)
    cv2.destroyAllWindows()

    old_corners = cv2.goodFeaturesToTrack(old_gray, **feature_params)
    initial_corners = old_corners[:, 0, :]
    displacement = np.zeros(len(old_corners))
    mask = np.zeros_like(frames[0])
    colors = np.random.randint(0, 255, (100, 3))
    new_frames_scene = []
    not_detected_counts = 0

    for i, frame in tqdm(enumerate(frames[1:])):
        new_gray = preprocess_frame(frame)
        new_corners, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_corners, None, **lk_params)
        displacement += (np.linalg.norm(new_corners[:, 0, :] - initial_corners, axis=1) / recalculate_every_n_frame)
        # displacement = (displacement - displacement.min()) / (displacement.max() - displacement.min())
        # displacement = (displacement - displacement.mean()) / displacement.std()
        detected = False
        distances = []

        keypounts_in_danger = check_keypoints(rect, new_corners[np.logical_and(st.ravel(),
                                                                               (displacement > min_displacement).ravel()), 0, :])
        if keypounts_in_danger > max_keypoints_allowed:
            detected = True

        # for j, (new, old) in enumerate(zip(new_corners[:, 0, :], old_corners[:, 0, :])):
        #     a, b = new.ravel()
        #     c, d = old.ravel()
        #     # distances.append(np.linalg.norm(new - old, 2))
        #     # dist = np.linalg.norm(new - old, 2)
        #     if st[j] and displacement[j] > min_displacement:
        #         # mask = cv2.line(mask, (a, b), (c, d), colors[j].tolist(), 2)
        #         frame = cv2.circle(frame, (a, b), 5, colors[j].tolist(), -1)

        # if i % 100 is 0:
        #     plt.hist(displacement)
        #     plt.show()
            # print(new_corners[])

        if detected:
            cv2.putText(frame, "Obstacle detected", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Obstacle not detected", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame = cv2.add(frame, mask)
        new_frames_scene.append(frame)
        cv2.imshow("frame", frame)

        if cv2.waitKey(10) & 0xff == ord("q"):
            break

        elif cv2.waitKey(30) & 0xff == ord("s"):
            cv2.imwrite(f"results1/{time()}.png", frame)

        old_gray = new_gray.copy()
        if i % recalculate_every_n_frame == 1:
            old_corners = cv2.goodFeaturesToTrack(new_gray, **feature_params)
            initial_corners = old_corners[:, 0, :]
            displacement = np.zeros(len(old_corners))

        else:
            old_corners = new_corners.reshape((-1, 1, 2))

