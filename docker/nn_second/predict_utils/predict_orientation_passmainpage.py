import numpy as np
from predict_utils.classificator_cnn_tf23 import Classifier, M

orientation = Classifier(M.ORIENTATION_PASSPORT)  # passport 1 # verty bad at pts!!
main_or_not = Classifier(M.PASSPORT_MAIN)  # passport 2


def predict(image) -> tuple:
    # return 0, 0
    img_rot1, img_rot2, img_rot3 = Classifier.rotate(image)  # clockwise
    #
    re_r0 = orientation.predict(image)
    re_r1 = orientation.predict(img_rot1)
    re_r2 = orientation.predict(img_rot2)
    re_r3 = orientation.predict(img_rot3)
    del img_rot1, img_rot2, img_rot3
    r_rot = Classifier.calc_orientation(re_r0[1], re_r1[1], re_r2[1], re_r3[1])

    # second try
    # y, x = img_rot1.shape
    # print(img_rot1.shape)
    # res_img = img_rot1[100:(y-100), 100:(x-100)]
    # print(res_img.shape)
    # res_img = imutils.resize(res_img, siz, siz)
    # print(res_img.shape)

    # FIX ORIENTATION
    if r_rot != 0:
        for _ in range(r_rot):
            image = np.rot90(image)  # not clockwise


    # MAIN OR NOT
    re_m = main_or_not.predict(image)

    return (r_rot, int(re_m[0]))
