import cv2 as cv
import imutils
# own
from cnn.shared_image_functions import crop, fix_angle, get_lines_c
from cnn.classes import siz

pass_photo_pts_size = siz // 2 // 2


# def rotatecv(img):
#     timg = cv.transpose(img)  # clockwise
#     img_rot1 = cv.flip(timg, flipCode=1)
#
#     timg = cv.transpose(img_rot1)  # clockwise
#     img_rot2 = cv.flip(timg, flipCode=1)
#
#     timg = cv.transpose(img_rot2)  # clockwise
#     img_rot3 = cv.flip(timg, flipCode=1)
#     return img_rot1, img_rot2, img_rot3


def prepare(img, rate=1) -> tuple:
    """
    For passport prediction and all classes prediction
    :param img: must b
    :param rate:
    :return: resized cropped gray, not cropped resized angle fixed gray, bgr_cropped_not_resized
    """
    # img = fix_angle(img, get_lines_c)  # fix angles for passport - done in GetDocumentType.__init__
    bgr_not_resized_cropped, gray = crop(img, rotate=True, rate=rate)
    # bgr_not_resized_cropped, gray = crop(img, rotate=True, rate=rate)
    resized = cv.resize(bgr_not_resized_cropped, (siz, siz))
    resized = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)  # TODO this
    # GRAY
    if gray.shape[0] > gray.shape[1]:  # by smallest
        gray = imutils.resize(gray, width=round(siz * 1.7))
    else:
        gray = imutils.resize(gray, height=round(siz * 1.7))
    gray = fix_angle(gray, get_lines_c)
    gray = cv.resize(gray, (pass_photo_pts_size, pass_photo_pts_size))

    return resized, gray, bgr_not_resized_cropped