import cv2
import os
import imutils

from helpers.face_helper import get_faces


class TestSetGenerator:
    # BGR color constants
    WHITE = (255, 255, 255)
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLACK = (0, 0, 0)

    def __init__(self, image_path):
        prototxt = "./ml_artifacts/deploy.prototxt.txt"
        model = "./ml_artifacts/res10_300x300_ssd_iter_140000.caffemodel"

        self.image_path = image_path
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)

    def capture_faces(self):
        frame = cv2.imread(self.image_path)
        frame = imutils.resize(frame, width=400)
        faces = get_faces(frame, self.net)
        frame_color = self.BLUE

        for (startX, startY, endX, endY) in faces:
            face_frame = frame[startY:endY, startX:endX]
            save_image(face_frame, "faces")


def save_image(image, folder):
    """Save an image with unique name
    Arguments:
        image {OpanCV} -- image object to be saved
        folder {string} -- output folder
    """

    # check whether the folder exists and create one if not
    if not os.path.exists(folder):
        os.makedirs(folder)

    # to not erase previously saved photos counter (image name) = number of photos in a folder + 1
    image_counter = len([name for name in os.listdir(folder)
                         if os.path.isfile(os.path.join(folder, name))])

    # increment image counter
    image_counter += 1

    # save image to the dedicated folder (folder name = label)
    cv2.imwrite(folder + '/' + str(image_counter) + '.png', image)