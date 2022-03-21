import cv2
import os
import numpy as np

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # resize without distortion
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    # return the resized image
    return resized


def convert_to_opencv(image):
    # RGB -> BGR conversion is performed as well.
    image = image.convert('RGB')
    r, g, b = np.array(image).T
    opencv_image = np.array([b, g, r]).transpose()
    return opencv_image


def crop_center(img, cropx, cropy):
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def resize_down_to_1600_max_dim(image):
    h, w = image.shape[:2]
    if (h < 1600 and w < 1600):
        return image

    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)


def resize_to_256_square(image):
    h, w = image.shape[:2]
    try:
        resized_image = cv2.resize(
            image, (256, 256), interpolation=cv2.INTER_LINEAR)
    except:
        resized_image = image
    return resized_image


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