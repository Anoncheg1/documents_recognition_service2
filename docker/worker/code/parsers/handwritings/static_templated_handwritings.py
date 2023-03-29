import numpy as np
import cv2 as cv
import inject
import os
# own
from logger import logger as log
from parsers.kaze_cropper import KazeCropper
from utils.coordinates_of_red_rectangles import get_market_rectange
from parsers.handwritings.utils_handwr import crop_to_list_of_squares

DSIZE_FOR_ALL_DOCS = (1060, 1500)

croppers: list = [None for _ in range(10)]
curr_dir = os.path.dirname(os.path.abspath(__file__))
p = curr_dir + '/handwriting_fields_templates/2.png'
o_i = cv.imread(p, cv.IMREAD_GRAYSCALE)
o_i = cv.resize(o_i, dsize=DSIZE_FOR_ALL_DOCS, interpolation=cv.INTER_CUBIC)
# import matplotlib.pyplot as plot
# plot.imshow(o_i)
# plot.show()

croppers[2] = KazeCropper(o_i)
p = curr_dir + '/handwriting_fields_templates/4.png'
o_i = cv.imread(p, cv.IMREAD_GRAYSCALE)
o_i = cv.resize(o_i, dsize=DSIZE_FOR_ALL_DOCS, interpolation=cv.INTER_CUBIC)
croppers[4] = KazeCropper(o_i)
p = curr_dir + '/handwriting_fields_templates/7.png'
o_i = cv.imread(p, cv.IMREAD_GRAYSCALE)
o_i = cv.resize(o_i, dsize=DSIZE_FOR_ALL_DOCS, interpolation=cv.INTER_CUBIC)
croppers[7] = KazeCropper(o_i)
p = curr_dir + '/handwriting_fields_templates/9p3.png'
o_i = cv.imread(p, cv.IMREAD_GRAYSCALE)
o_i = cv.resize(o_i, dsize=DSIZE_FOR_ALL_DOCS, interpolation=cv.INTER_CUBIC)
croppers[9] = KazeCropper(o_i)

roi_coordinates: list = [None for _ in range(10)]
p = curr_dir + '/handwriting_fields_templates/2_r.png'
o_i = cv.imread(p, cv.IMREAD_COLOR)
o_i = cv.resize(o_i, dsize=DSIZE_FOR_ALL_DOCS)
roi_coordinates[2] = get_market_rectange(o_i)
p = curr_dir + '/handwriting_fields_templates/4_r.png'
o_i = cv.imread(p, cv.IMREAD_COLOR)
o_i = cv.resize(o_i, dsize=DSIZE_FOR_ALL_DOCS)
roi_coordinates[4] = get_market_rectange(o_i)
p = curr_dir + '/handwriting_fields_templates/7_r.png'
o_i = cv.imread(p, cv.IMREAD_COLOR)
o_i = cv.resize(o_i, dsize=DSIZE_FOR_ALL_DOCS)
roi_coordinates[7] = get_market_rectange(o_i)
p = curr_dir + '/handwriting_fields_templates/9p3_r.png'
o_i = cv.imread(p, cv.IMREAD_COLOR)
o_i = cv.resize(o_i, dsize=DSIZE_FOR_ALL_DOCS)
roi_coordinates[9] = get_market_rectange(o_i)

PREDICT_SIZE = round(576 // 2 // 2)

# supported:
PAGES = {9: [3]}


@inject.params(predict='predict_is_handwrited')
def parse_handwritings_static(img: np.ndarray, doc_type: int, page: int = 1, predict=None) -> dict or None:
    """
    :param img: no side effect
    :param doc_type: int 2, 4, 7, 9
    :param page:
    :param predict:
    :return:
    """
    # check if not supported
    if doc_type in PAGES.keys() and page not in PAGES[doc_type]:
        return None

    im = cv.resize(img, dsize=DSIZE_FOR_ALL_DOCS)
    ret_result = {}
    # {'signature1': None,
    #               'signature2': None,
    #               'some_text_field': None,
    #               'FIO': None}
    # orientation
    cr: KazeCropper = croppers[doc_type]
    img_r = cr.crop(im, double_crop=False)
    if img_r is None:
        log.warning("cannot apply cropper for page {}, doctype {}".format(page, doc_type))
        return None
    recs = roi_coordinates[doc_type]

    def predict_and_set_ret_bool_macros(rect_number: int or tuple, recs: list, what: str or tuple):
        if type(rect_number) is int:
            x, y, w, h = recs[rect_number]
            roi = img_r[y:y + h, x:x + w]
            roi = cv.resize(roi, (PREDICT_SIZE, PREDICT_SIZE))
            ret_result[what] = bool(predict((roi,))[0])
        else:
            rois = []
            for rect_num in rect_number:
                x, y, w, h = recs[rect_num]
                roi = img_r[y:y + h, x:x + w]
                roi = cv.resize(roi, (PREDICT_SIZE, PREDICT_SIZE))
                # import matplotlib.pyplot as plot
                # plot.imshow(roi)
                # plot.show()
                rois.append(roi)
            hw_pred: tuple = predict(tuple(rois))
            for wt, p in zip(what, hw_pred):
                ret_result[wt] = bool(p)

    def predict_macros(rect_numbers: tuple, recs: list) -> int:
        rois = []
        for n in rect_numbers:
            x, y, w, h = recs[n]
            roi = img_r[y:y + h, x:x + w]
            roi = cv.resize(roi, (PREDICT_SIZE, PREDICT_SIZE))
            rois.append(roi)
        return predict(tuple(rois))

    def long_line_predict(rect_number: int, recs: list, what: str):
        x, y, w, h = recs[rect_number]
        roi = img_r[y:y + h, x:x + w]

        rois = crop_to_list_of_squares(roi)
        assert rois

        rois = (cv.resize(r, (PREDICT_SIZE, PREDICT_SIZE), interpolation=cv.INTER_CUBIC) for r in rois)
        r = predict(tuple(rois))
        # if what == 'FIO':
        #     print("r", r)
        if any(r):
            ret_result[what] = True
        else:
            ret_result[what] = False

    if doc_type == 2:
        recs = sorted(recs, key=lambda y: y[1])
        predict_and_set_ret_bool_macros(0, recs, 'signature1')
    elif doc_type == 4:
        recs = sorted(recs, key=lambda y: y[1])
        sig1, sig2 = predict_macros((1, 2), recs)
        if sig1 is True or sig2 is True:
            ret_result['signature1'] = True
        else:
            ret_result['signature1'] = False
    elif doc_type == 7:
        recs = sorted(recs, key=lambda x: x[0])
        predict_and_set_ret_bool_macros((0, 1), recs, ('signature1', 'signature2'))
    elif doc_type == 9 and page == 3:
        recs = sorted(recs, key=lambda y: y[1])
        # 0 - handwr
        # 1 - sig1
        # 2 - fio
        # 3 - sig2
        # sig 1 sig 2
        long_line_predict(1, recs, 'signature1')
        predict_and_set_ret_bool_macros(3, recs, 'signature2')
        # handwr
        long_line_predict(0, recs, 'some_text_field')

        # fio
        long_line_predict(2, recs, 'FIO')

    # -- debug --:
    #
    # for i, rec in enumerate(recs):
    #     x, y, w, h = rec
    #     cv.rectangle(img_r, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
    #     cv.putText(img_r, str(i),
    #                org=(x + i * 40, y - 10),
    #                fontFace=cv.FONT_HERSHEY_PLAIN,
    #                fontScale=3,
    #                color=(0, 255, 0),
    #                thickness=2)
    # #
    # tmp = cv.resize(img_r, (900, 900))
    # cv.imshow('image', tmp)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q
    # tmp = cv.resize(roi1, (300, 300))
    # cv.imshow('image', tmp)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q
    # -- the end --

    if any([x is not None for x in ret_result.values()]):
        return ret_result
    else:
        return None


def test_static_handwriting():
    # from test import profiling_before, profiling_after
    from predict_utils.classificator_cnn_tf23 import Classifier

    Classifier.save_dir = '/home/u2/h4/PycharmProjects/rec2/selected_models'

    from predict_utils.predict_handwriting_or_not import predict

    inject.clear_and_configure(lambda binder: binder
                               .bind_to_provider('predict_is_handwrited', lambda: predict))

    # p = '/home/u2/h4/signature_templates/2/PNG0.png'
    # p = '/home/u2/h4/signature_templates/4/PNG0.png'
    # p = '/home/u2/h4/signature_templates/7/PNG0.png'
    # p = '/home/u2/h4/signature_templates/9/PNG0.png'
    # p = '/home/u2/h4/signature_templates/9/PNG2.png'
    p = '/home/u2/h4/signature_templates/9/PNG2_lower.png'
    # p = '/home/u2/h4/PycharmProjects/rec2/test/handwrited_text_9.png'
    doc_type = 9
    img = cv.imread(p)
    # pr = profiling_before()
    img_save = img.copy()
    print(parse_handwritings_static(img, doc_type, page=3))
    print((img_save == img).all())
    # profiling_after(pr)  # noqa


if __name__ == '__main__':
    test_static_handwriting()

