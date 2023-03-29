import numpy as np
# own
from predict_utils.classificator_cnn_tf1 import Classifier, M

pass_photo_pts = Classifier(M.PASS_PHOTO_PTS)  # 1
# p_without_vd = Classifier(M.P_WITHOUT_VD)  # 2


def predict(image) -> tuple:  # return int
    """
    from predict_utils.cvutils import prepare
    (_, this, _) = prepare(image, rate=1)
    :param image:
    :return:
    """
    # return (4,)
    gray_rot1, gray_rot2, gray_rot3 = Classifier.rotate(image)  # clockwise
    re0 = pass_photo_pts.predict(image)
    re1 = pass_photo_pts.predict(gray_rot1)
    re2 = pass_photo_pts.predict(gray_rot2)
    re3 = pass_photo_pts.predict(gray_rot3)
    av_logits = np.average((re0[1], re1[1], re2[1], re3[1]), axis=0)  # passp, photo, pts, vd
    pass_photo_pts_re = int(np.argmax(av_logits, axis=-1))

    # if pass_photo_pts_re == 0:
    #     re0_2 = p_without_vd.predict(image)
    #     re1_2 = p_without_vd.predict(gray_rot1)
    #     re2_2 = p_without_vd.predict(gray_rot2)
    #     re3_2 = p_without_vd.predict(gray_rot3)
    #     av_logits_2 = np.average((re0_2[1], re1_2[1], re2_2[1], re3_2[1]), axis=0)  # passport alone or p + vd
    #     p_without_vd_re = int(np.argmax(av_logits_2, axis=-1))
    #     # print(p_without_vd_re)
    #     if p_without_vd_re == 1:
    #         pass_photo_pts_re = 4

    # print(re0)
    # print(re1)
    # print(re2)
    # print(re3)
    # del image, gray_rot1, gray_rot2, gray_rot3



    # print(re0[1])
    # print(av_logits)

    # re = av_logits - np.mean((av_logits[0], av_logits[1], av_logits[2], av_logits[2]))  # without last it is better
    #
    # if re[0] > 0 and re[1] < 0 and re[2] < 0 and re[3] > 0 \
    #         and re[0] > re[3] and re[3] * 2 > re[0]:  # passport_and_vod  # re[0] > re[3]
    #     pass_photo_pts_re = 4
    # else:
    #     pass_photo_pts_re = int(np.argmax(av_logits, axis=-1))  # max


    return (pass_photo_pts_re,)

