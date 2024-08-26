# pylint: disable=no-member
from pathlib import Path
from time import time
import os
import sys
import cv2



def main(video_path, output_path):

    capture = cv2.VideoCapture(video_path)

    video_dir = Path(video_path)
    image_dir = Path(video_path).stem + "_imgs"
    image_dir = Path(output_path) / image_dir
    image_dir.mkdir(exist_ok=True, parents=True)

    fps = capture.get(cv2.CAP_PROP_FPS)
    print("FPS", fps, video_dir)
    start = time()

    frames_done = sorted(
        [int(i.split('_')[1].split('.')[0]) for i in os.listdir(image_dir) if i.endswith('.jpg')])

    if frames_done:
        frames_done = frames_done[-1]
    else:
        frames_done = 0

    frame_count = 0
    while frame_count < 1_000_000:
        if frame_count < frames_done:
            frame_count += 1
            continue

        success, frame = capture.read()
        if not success:
            break

        if frame_count % (fps * 2) == 0:
            print(frame_count)
            cv2.imwrite(str(image_dir / f'frame_{frame_count}.jpg'), frame)

        frame_count += 1

    capture.release()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
    print("DONE!")
