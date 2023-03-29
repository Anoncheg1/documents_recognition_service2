import numpy as np
import cv2 as cv
import inject
# own
from logger import logger as log
from parsers.handwritings.static_templated_handwritings import DSIZE_FOR_ALL_DOCS
from parsers.handwritings.variable_fix_orientation import fix_angle_txtdoc
PREDICT_SIZE = round(576 // 2 // 2)


@inject.params(predict='predict_is_handwrited')
def parse_variable_fields(image: np.array, doc_type: int, page: int, pages=None, predict=None) -> dict or None:
    """
    :param image: no side effect
    :param doc_type: 3, 6, 8
    :param page:
    :param pages:
    :param predict: not used, injecting
    :return: singature1 signature2 signature3
    """
    ret_result = {}
    # {'signature1': None,
    #               'signature2': None,
    #               'signature3': None
    #               }

    def predict_and_set_ret_bool_macros(img: np.array, rect: tuple, what: str):
        # global PREDICT_SIZE, predict, ret_result
        x, y, w, h = rect
        roi = img[y:y + h, x:x + w]  # resized BGR
        top = int(0.5 * roi.shape[0])
        bottom = top
        left = int(0.5 * roi.shape[0])
        right = left
        roi = cv.copyMakeBorder(roi, top, bottom, left, right, cv.BORDER_REFLECT)

        # cv.imwrite("/home/u2/tmp2.png", roi)
        roi = cv.resize(roi, (PREDICT_SIZE, PREDICT_SIZE))
        ret_result[what] = bool(predict((roi,))[0])
        # tmp = cv.resize(roi, (500, 600))
        # cv.imshow('image', tmp)  # show image in window
        # cv.waitKey(0)  # wait for any key indefinitely
        # cv.destroyAllWindows()  # close window q

    image = cv.resize(image, DSIZE_FOR_ALL_DOCS)
    image = fix_angle_txtdoc(image)
    try:
        img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    except:  # noqa
        img = image
    otsu = cv.threshold(
        img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    binary = cv.threshold(
        img, np.mean(img), 255, cv.THRESH_BINARY_INV)[1]

    img = otsu + binary

    # remove noise
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel=np.ones((2, 2)), iterations=3)
    # group contours
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel=np.ones((20, 20)), iterations=2)

    # -- find barcode
    treshold_area_min = 7000
    treshold_area_max = 13000
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    barcode_cnt = None
    barcode_min = None
    for cnt in contours:  # cnt = contours[0]
        area = cv.contourArea(cnt)
        y = cnt[0][0][1]
        if treshold_area_max > area > treshold_area_min and y > 1000:
            # print("bar",area, min([x[0][1] for x in cnt]))
            b_min =min([x[0][1] for x in cnt])
            if barcode_cnt is None or b_min > barcode_min:
                barcode_min = b_min
                barcode_cnt = cnt

    if barcode_cnt is None:
        log.warning("Can not find barcode")
        return


    # -- find lowest text
    lowest_text_cnts = []
    lowest_text_y = None
    barcode_highest_y = min([x[0][1] for x in barcode_cnt])
    # print("barcode_highest_y", barcode_highest_y)
    treshold_area_min = 10
    # print(barcode_highest_y, [x[0][1] for x in barcode_cnt])
    for cnt in contours:  # cnt = contours[0]
        area = cv.contourArea(cnt)
        y = max([x[0][1] for x in cnt])
        if area > treshold_area_min and barcode_highest_y > y:
            lowest_text_cnts.append(cnt)
    if len(lowest_text_cnts) == 0:
        log.warning("Can not find lowest text")
        return
    lowest_text_y = max([x[0][1] for c in lowest_text_cnts for x in c])

    if lowest_text_y is None:
        log.warning("Can not find lowers_text_y")
        return

    # -- select rectangles
    if doc_type == 3 and page != pages:
        x = 40
        y = lowest_text_y - 100
        w = 300
        h = 150
        sign_rect1 = x, y, w, h
        x = 500
        sign_rect2 = x, y, w, h
        predict_and_set_ret_bool_macros(image, sign_rect1, 'signature1')
        predict_and_set_ret_bool_macros(image, sign_rect2, 'signature2')
    elif doc_type == 3 and page == pages:
        x = 50
        y = lowest_text_y - 150
        w = 200
        h = 70
        sign_rect1 = x, y, w, h
        x = 530
        w = 170
        sign_rect2 = x, y, w, h
        x = 750
        y = lowest_text_y - 90
        sign_rect3 = x, y, w, h
        predict_and_set_ret_bool_macros(image, sign_rect1, 'signature1')
        predict_and_set_ret_bool_macros(image, sign_rect2, 'signature2')
        predict_and_set_ret_bool_macros(image, sign_rect3, 'signature3')
    elif doc_type == 6 and page == pages:
        x = 40
        y = lowest_text_y - 230
        w = 300
        h = 120
        sign_rect1 = x, y, w, h
        x = 600
        sign_rect2 = x, y, w, h
        predict_and_set_ret_bool_macros(image, sign_rect1, 'signature1')
        predict_and_set_ret_bool_macros(image, sign_rect2, 'signature2')
    elif doc_type == 8 and page == pages:
        x = 40
        y = lowest_text_y - 100
        w = 300
        h = 150
        sign_rect1 = x, y, w, h
        x = 500
        sign_rect2 = x, y, w, h
        predict_and_set_ret_bool_macros(image, sign_rect1, 'signature1')
        predict_and_set_ret_bool_macros(image, sign_rect2, 'signature2')

    # DEBUG
    # image = np.zeros(image.shape)
    # cv.drawContours(image, lowest_text_cnts, -1, color=(0, 255, 0), thickness=5)
    # x, y, w, h = sign_rect1
    # cv.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=5)
    # x, y, w, h = sign_rect2
    # cv.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=5)
    # x, y, w, h = sign_rect3
    # cv.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=5)
    # #
    # import matplotlib.pyplot as plot
    # plot.imshow(img)
    # plot.show()
    # #
    #
    # tmp = cv.resize(image, (500, 600))
    # cv.imshow('image', tmp)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q

    if any([x is not None for x in ret_result.values()]):
        return ret_result
    else:
        return None


def _test_make_border():
    p = '/home/u2/tmp2.png'
    img = cv.imread(p, cv.IMREAD_COLOR)

    top = int(0.5 * img.shape[0])
    bottom = top
    left = int(0.5 * img.shape[0])
    right = left
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_REFLECT)

    print(img.shape)
    tmp = cv.resize(img, (500, 600))
    cv.imshow('image', tmp)  # show image in window
    cv.waitKey(0)  # wait for any key indefinitely
    cv.destroyAllWindows()  # close window q


def _test_parser():

    from predict_utils.classificator_cnn_tf23 import Classifier

    Classifier.save_dir = '/home/u2/h4/PycharmProjects/rec2/selected_models'

    from predict_utils.predict_handwriting_or_not import predict

    inject.clear_and_configure(lambda binder: binder
                               .bind_to_provider('predict_is_handwrited', lambda: predict))
    # from utils.profiling import profiling_before, profiling_after
    # p = '/home/u2/h4/signature_templates/3/i/PNG1.png'
    # p = '/home/u2/h4/signature_templates/3/i/PNG5.png'
    p = '/home/u2/h4/signature_templates/3/i2/PNG5_lower.png'
    # p = '/home/u2/h4/signature_templates/6/PNG0.png'
    # p = '/home/u2/h4/signature_templates/8/PNG0.png'
    doc_type = 3
    img = cv.imread(p, cv.IMREAD_COLOR)
    # pr = profiling_before()
    img_save = img.copy()
    print(parse_variable_fields(img, doc_type, page=5, pages=5))  # , doc_type, page=3
    print((img_save == img).all())
    # profiling_after(pr)


if __name__ == '__main__':
    _test_parser()
    # test_make_border()
