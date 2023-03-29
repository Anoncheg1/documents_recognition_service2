import cv2 as cv
from typing import Callable
import numpy as np
# own
from cnn.shared_image_functions import get_lines_canny
from cnn.shared_image_functions import img_to_small, rotate_detect, rotate


def fix_angle_txtdoc(img_orig, gl: Callable = get_lines_canny) -> np.array:  # , copy=None
    """ Fix little angles
    1) image to 575 by width
    3) rotate image by degrees and find out angles with gl:Callable for every degree

    :param img_orig: no side effect
    :param gl:
    :return: image
    """
    img_small, _ = img_to_small(img_orig, height_target=575)
    # cv.imshow('image', img_small)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window
    try:
        img_small = cv.cvtColor(img_small, cv.COLOR_BGR2GRAY)
    except:  # noqa
        pass
    median = np.median(rotate_detect(img_small, gl))
    # print(median)

    # rotate
    # FINAL ROATE
    if abs(median) > 1:
        ret_img = rotate(img_orig, median)
    else:  # no blur of warpAffine
        ret_img = img_orig
    return ret_img


if __name__ == '__main__':  # test
    import random
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import tensorflow as tf

    # from test import profiling_before, profiling_after

    def add_noise(img):
        """Add random noise to an image"""
        VARIABILITY = 2
        deviation = VARIABILITY * random.random()
        print(deviation)
        noise = np.random.normal(0, deviation, img.shape)
        img += noise
        np.clip(img, 0., 255.)
        return img


    image_gen = ImageDataGenerator(# featurewise_center=True,
                                   # featurewise_std_normalization=True,
                                   # samplewise_std_normalization=True,
                                   rotation_range=25,
                                   shear_range=0.10,
                                   zoom_range=[0.90, 1.10],
                                   # horizontal_flip=True,
                                   # vertical_flip=False,
                                   # data_format='channels_last',
                                   brightness_range=[0.6, 1.3],
                                   # zca_epsilon=1e-6,
                                   channel_shift_range=0.1,
                                   dtype=tf.dtypes.int8,
                                   preprocessing_function=add_noise
                                   )

    p = '/home/u2/h4/PycharmProjects/cnn/text_or_not/samples/text/signed/'

    # p = '/home/u2/h4/signature_templates/3/i/PNG2.png'
    import os
    files = [file for file in os.listdir(p) if file.lower().endswith('png')]
    for i, file in enumerate(files):
        if i < 1:
            continue
        print(i, file)
        file_p = os.path.join(p, file)
        img = cv.imread(file_p)  #, cv.IMREAD_GRAYSCALE)
        if img is None:
            print("fail to read"), exit()

        # noise and rotate
        # img2 = img.reshape((1,) + img.shape + (1,))
        # img_ch = image_gen.flow(img2, y=None, shuffle=False)[0]
        # img_ch = img_ch.reshape(img.shape)
        # img_ch = img_ch.astype(np.uint8)
        # img = img_ch

        # pr = profiling_before()
        img_save = img.copy()
        im = fix_angle_txtdoc(img, get_lines_canny)
        print((img_save == img).all())
        # profiling_after(pr)
        # lines = get_lines_canny(img)
        #
        # ret = rotate_detect(img, get_lines_canny)
        # print(ret)
        #
        # median = ret
        # img_orig = img
        # if abs(median) > 1:
        #     scale = 1
        #      ret_img = rotate(img_orig, median, scale=scale)
        # else:  # no blur of warpAffine
        #     ret_img = img_orig
        #
        # # print("LINES!!!!!!!", len(lines))
        # img2 = img.copy()
        # for line in lines:
        #     x1, y1, x2, y2 = line[0]
        #     cv.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 3)  # DEBUG!!
        # #
        # img2 = cv.resize(img2, (900, 900))
        # # plt.imshow(img)
        # # plt.show()
        # cv.imshow('image', img2)  # show image in window
        # cv.waitKey(0)  # wait for any key indefinitely
        # cv.destroyAllWindows()  # close window

        # tmp = cv.resize(ret_img, (900, 900))
        tmp = cv.resize(img, (400, 400))
        tmp2 = cv.resize(im, (400, 400))
        tmp3 = np.concatenate((  tmp, tmp2), axis=1)
        cv.imshow('image', tmp3)  # show image in window
        cv.waitKey(0)  # wait for any key indefinitely
        cv.destroyAllWindows()  # close window