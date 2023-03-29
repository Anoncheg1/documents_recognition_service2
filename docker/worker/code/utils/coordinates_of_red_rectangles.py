import cv2 as cv
import numpy as np
from typing import List, Tuple


def get_market_rectange(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """ TODO: sort rectangles to ensure list are fixed
    :param img: cv.IMREAD_COLOR BGR
    :return x, y, w, h - int"""
    # img = cv.imread(p, cv.IMREAD_COLOR)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # red rectangle hsv filter
    lower_red = np.array([0, 10, 10])
    upper_red = np.array([10, 255, 255])
    mask = cv.inRange(img_hsv, lower_red, upper_red)
    img_with_squares = cv.bitwise_and(img, img, mask=mask)
    # print(img_with_squares.shape)
    # print(img_with_squares.dtype)

    img_with_squares = cv.cvtColor(img_with_squares, cv.COLOR_HSV2BGR)
    img_with_squares = cv.cvtColor(img_with_squares, cv.COLOR_BGR2GRAY)
    contours, hierarchy = cv.findContours(img_with_squares, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # get rectangle points:
    ret = []
    for cnt in contours:  # cnt = contours[0]
        x, y, w, h = cv.boundingRect(cnt)
        # remove red rectangle
        x += 1
        y += 1
        w -= 2
        h -= 2
        # roi = img[y:y + h, x:x + w]
        # cv.drawContours(img, contours, 0, (0, 255, 0))
        # return roi  # BGR
        ret.append((x, y, w, h))
    return ret


if __name__ == '__main__':  # test
    p = '/home/u2/Downloads/examples/2/obr3_r.png'
    roi = get_market_rectange(p, 2)
    # tmp = imutils.resize(img, height=800)  # resized
    # cv.imshow('image', tmp)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q
    #
    import imutils
    tmp = imutils.resize(roi, height=800)  # resized
    cv.imshow('image', tmp)  # show image in window
    cv.waitKey(0)  # wait for any key indefinitely
    cv.destroyAllWindows()  # close window q

    # green = np.uint8([[[85,255,255]]])
    # green = cv.cvtColor(green, cv.COLOR_HSV2RGB)
    # print(green)
    # lower_red = np.array([0])
    # cv.inRange()
