"""Face Detection with opencv DNN model
"""
import os
from multiprocessing import Pool
from multiprocessing import cpu_count
from imutils.face_utils import FaceAligner
import dlib
import cv2
import numpy as np

def face_detection(file):
    """Detect face using GPU resource

    Arguments:
        file -- input image name
    """
    # Loop through all images and save images with marked faces
    _, file_extension = os.path.splitext(file)

    if (file_extension in ['.png', '.jpg', '.jpeg']):
        image = cv2.imread(data_path + file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (h, w) = image.shape[:2]

        # Blob detector
        ## The last arguments of blobFromImage is the mean value of RGB
        ## It might shows good performance without change the value
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        model.setInput(blob)
        detections = model.forward()
        # detections[0, 0, :, 2] -> confidence
        # detections[0, 0, :, 3:7] -> (startX, startY, endX, endY)

        # Detect face with highest confidence
        box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])

        # Set coordinate
        (startX, startY, endX, endY) = box.astype("int")
        rect = dlib.rectangle(startX, startY, endX, endY)

        if alignment and crop:
            # crop and alignment
            image = fa.align(image, gray, rect)
        else:
            if crop:
                # just cropping
                image = image[startY:endY, startX:endX]
            else:
                # Draw bounding box & save the image
                cv2.rectangle(image, (startX, startY), (endX, endY), (255, 255, 255), 2)

        cv2.imwrite(os.path.join(save_path, file), image)
        # cv2.imshow("faces", image)
        # cv2.waitKey()
        print("Image " + file + " converted successfully")

def run(imgs):
    """Function for multiprocessing

    Arguments:
        imgs {str} - image name
    """
    print("[INFO] Launching pool using {} processes...".format(cpu_count()-1))
    pool = Pool(cpu_count()-1)
    pool.map(face_detection, imgs)

    print("[INFO] Waiting for processes to finish...")
    pool.close()
    pool.join()
    print("[INFO] Multiprocessing complete")

# Use CelebA dataset for testing
# You can download the dataset on the following url:
# https://drive.google.com/drive/folders/1lENOECdd-8is7RnVSGOrlCdamYJ8hyhd
# CelebA > Img > img_celeba.7z
# We use first 100 images in CelebA for test face detection performance

if __name__ == '__main__':
    # Define paths
    base = os.path.dirname(os.path.dirname(__file__))
    current = os.path.dirname(__file__)
    prototxt_path = os.path.join(current, 'model_data/deploy.prototxt')
    model_path = os.path.join(current, 'model_data/res10_300x300_ssd_iter_140000_fp16.caffemodel')
    aligner_path = os.path.join(current, 'model_data/shape_predictor_68_face_landmarks.dat')

    # You need to change following path
    data_path = os.path.join(base, 'celeba/non_aligned/')
    save_path = os.path.join(base, 'celeba/gpu/')

    # create new folder (if it does not exis)
    os.makedirs(save_path, exist_ok=True)

    # Set model with pretrained weights
    model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Set gpu
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # Set aligner
    predictor = dlib.shape_predictor(aligner_path)
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    # call data and run main function
    img_list = os.listdir(data_path)

    # You need to change following 'crop' and 'alignment' for your purpose
    crop = True
    alignment = True

    run(img_list)
