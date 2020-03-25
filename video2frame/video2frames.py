"""Convert videos to frames
"""
import os
import cv2
import tqdm

def splitFrames(video_path, save_path):
    """split frames from video and save them

    Arguments:
        video_path {string} -- The location of videos
        save_path {string} -- save path
    """

    sub = os.listdir(video_path)[0]

    # check whether your source dir has "sub dir" or not (just include videos)
    if os.path.isdir(os.path.join(video_path, sub)):
        folders = os.listdir(video_path)

        for folder in tqdm.tqdm(folders):
            videos = os.listdir(os.path.join(video_path, folder))
            for video in videos:
                vidcap = cv2.VideoCapture(os.path.join(video_path, folder, video))
                success, image = vidcap.read()
                count = 0

                save_path_video = os.path.join(save_path, video)
                os.makedirs(save_path_video, exist_ok=True)
                while success:
                    cv2.imwrite("{}/frame{}.jpg".format(save_path_video, count), image)
                    success, image = vidcap.read()
                    count += 1

    else:
        videos = os.listdir(video_path)

        for video in tqdm.tqdm(videos):
            vidcap = cv2.VideoCapture(os.path.join(video_path, video))
            success, image = vidcap.read()
            count = 0

            save_path_video = os.path.join(save_path, video)
            os.makedirs(save_path_video, exist_ok=True)
            while success:
                cv2.imwrite("{}/frame{}.jpg".format(save_path_video, count), image)
                success, image = vidcap.read()
                count += 1


if __name__ == '__main__':
    # folder path
    source_path = 'your/data/path'
    target_path = 'save/path'

    splitFrames(source_path, target_path)
