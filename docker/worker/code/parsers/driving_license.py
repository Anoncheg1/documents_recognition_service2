import cv2 as cv
import imutils
import re
import pytesseract
from pytesseract import Output
from pytesseract.pytesseract import TesseractError
import inject
# own
from cnn.shared_image_functions import fix_angle, get_lines_c
from doc_types import DocTypeDetected
from parsers.passport_utils import razd_sub_re, find_date
from parsers.ocr import threashold_by_letters
from parsers.passport_mrz import re_spaces, letters_to_digits
from parsers.translit_drivingl import check
from parsers.driving_utils import compare_dates, check_dates, PravaCropper
from parsers.templates.loader import load
from logger import logger
from parsers import ocr
from utils.progress_and_output import OutputToRedis
from tesseract_lock import tesseract_image_to_data_lock, tesseract_image_to_string_lock

pcr = PravaCropper()

# test
# from groonga import FIOChecker
# fio_checker = FIOChecker(10)

fio_checker = inject.instance('FIOChecker')


class Anonymous:
    def __init__(self, msg: str = None, qc: int = None):
        """ Exception """
        if msg and qc:
            self.OUTPUT_OBJ: dict = {'qc': qc, 'exception': msg}
        else:
            """ Empty QC = 4 """
            self.OUTPUT_OBJ: dict = {'qc': 4}

# def reinject():  # for test
#     global fio_checker
#     fio_checker = inject.instance('FIOChecker')

# --- deprecated method ----
# t_rects = read_rectangles_template()
# t_rects = sorted(t_rects, key=lambda x: x[1], reverse=True)
# number_rect = t_rects[0]
# (x, y, w, h) = number_rect
# def read_rectangles_template() -> list:
#     """
#     not in work yeat
#     :return:
#     """
#     obr_old_v = '/home/u2/Desktop/obr_old_v4_test.png'
#     image = cv.imread(obr_old_v)
#     image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     # ret, image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
#     # cv.imshow('image', image)  # show image in window
#     # cv.waitKey(0)  # wait for any key indefinitely
#     # cv.destroyAllWindows()  # close window q
#     z = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
#     # ret, thresh = cv.threshold(image, 127, 255, 0)
#     (_, cnts, hierarchy) = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     # cnts2 = []
#     # for i in range(len(cnts)):
#     #     if hierarchy[0][i][3] == -1:
#     #         cnts2.append(c)
#     #         # cv2.drawContours(image_external, contours, i,
#     #         #                  255, -1)
#
#     boxes = []
#     for c in cnts:
#         box = cv.boundingRect(c)
#         x, y, w, h = box
#         cv.rectangle(z, (x, y), (x + w, y + h), (0, 255, 0), 1)  # debug
#         boxes.append(box)
#     # print(len(boxes))
#     # cv.imshow('image', z)  # show image in window
#     # cv.waitKey(0)  # wait for any key indefinitely
#     # cv.destroyAllWindows()  # close window q
#     return boxes


def _compare_vod_snum(one: str, two: str) -> str or None:
    if one is None and two:
        return razd_sub_re.sub('', two)
    elif one and two is None:
        return razd_sub_re.sub('', one)
    elif one and two:
        two = razd_sub_re.sub('', two)
        one = razd_sub_re.sub('', one)
        if len(one) == 10:
            return one
        elif len(two) == 10:
            return two
        elif len(one) != 0:
            return one
        elif len(two) != 0:
            return two
    return None


# def threashold_by_letters(gray, amount=160, scale_factor=2) -> np.ndarray or None:
#     """
#     :param scale_factor: must be 1 or 2 for passport and prava
#     :return: np.ndarray or None if letters_count < amount / 2
#     """
#     r_ret = None
#     r_max = 0
#     r_let = None
#     for i in range(10):  # recognition loop
#         try:
#             r = roi_preperation(gray, i, scale_factor=scale_factor)
#             letters = pytesseract.image_to_boxes(r, lang='rus')
#             letters = letters.split('\n')
#             letters = [letter.split() for letter in letters]
#             lc = len(letters)
#             # print(lc)
#             # >160 or max saved
#             if lc > amount:
#                 r_ret = r
#                 r_max = lc
#                 r_let = letters
#                 break
#             elif lc > r_max:
#                 r_ret = r
#                 r_max = lc
#                 r_let = letters
#         except TesseractError as ex:
#             logger.error("ERROR: %s" % ex.message)
#     if r_max < amount / 2:
#         return None, None
#     if r_let:
#         r_let = ''.join(list(zip(*r_let))[0])
#     return r_ret, r_let


# def threashold_by_letters(gray, amount=160) -> np.ndarray or None:
#     r_ret = None
#     r_max = 0
#
#     for i in range(10):  # recognition loop
#         try:
#             r = roi_preperation(gray, i, scale_factor=2)
#             letters = pytesseract.image_to_boxes(r, lang='rus')
#             letters = letters.split('\n')
#             letters = [letter.split() for letter in letters]
#             lc = len(letters)
#             # print(lc)
#             # >160 or max saved
#             if lc > amount:
#                 r_ret = r
#                 r_max = lc
#                 break
#             elif lc > r_max:
#                 r_ret = r
#                 r_max = lc
#         except TesseractError as ex:
#             logger.error("ERROR: %s" % ex.message)
#     if r_max < 3:
#         return None
#     return r_ret

def _crop_minAreaRect(img, rect):
    center, size, theta = rect
    if theta < -45:
        theta += 90
    (rows, cols) = img.shape[0], img.shape[1]
    M = cv.getRotationMatrix2D(center, theta, 1)
    img_rot = cv.warpAffine(img, M, (cols, rows))

    center, size = tuple(map(int, center)), tuple(map(int, size))
    out = cv.getRectSubPix(img_rot, size, center)

    return out


# side effect
def _suggest_dl(ret: dict):
    fio_sug = {
        'F': None,
        'F_gender': None,
        'F_score': 0,
        'I': None,
        'I_gender': None,
        'I_score': 0,
        'O': None,
        'O_gender': None,
        'O_score': 0,
    }
    res_i, res_o, res_f = None, None, None
    if ret['name_rus'] and len(ret['name_rus']) > 0:
        rets = ret['name_rus'].split()
        name = rets[0]
        res_i = fio_checker.wrapper_with_crop_retry(fio_checker.query_name, name)
    if ret['name_rus'] and len(ret['name_rus'].split()) > 1:
        patr_split = ret['name_rus'].split()[1:]
        res_o = fio_checker.wrapper_with_crop_retry(fio_checker.query_patronymic, ' '.join(patr_split))

    if ret['fam_rus'] and len(ret['fam_rus']) > 0:
        res_f = fio_checker.wrapper_with_crop_retry(fio_checker.query_surname, ret['fam_rus'])

    if res_i:
        (fio_sug['I'], fio_sug['I_gender'], fio_sug['I_score']) = res_i
    if res_o:
        (fio_sug['O'], fio_sug['O_gender'], fio_sug['O_score']) = res_o
    if res_f:
        (fio_sug['F'], fio_sug['F_gender'], fio_sug['F_score']) = res_f

    if res_i or res_o or res_f:
        ret['suggest'] = fio_sug


def parse_front_text(binary) -> dict or None:
    """ Get black text from main side of driving license
    and check by transliteration

    :param binary: image
    :return: return dictionary or None
    """
    # from matplotlib import pyplot as plt
    # plt.imshow(binary)
    # plt.show()

    try:
        with tesseract_image_to_data_lock:
            ocr_rus = pytesseract.image_to_data(binary, lang='rus', output_type=Output.DICT,
                                                config='-c tessedit_char_blacklist=' + ocr.rus_bad)

            ocr_eng = pytesseract.image_to_data(binary, lang='eng', output_type=Output.DICT,
                                                config='-c tessedit_char_blacklist=' + ocr.eng_bad)

    except TesseractError as ex:
        logger.error("ERROR: %s" % ex.message)
        return None

    ret: dict = {'fam_rus': None,
                 'fam_eng': None,
                 'fam_check': False,
                 'name_rus': None,  # name-one word and patronymic-may have several words
                 'name_eng': None,
                 'name_check': False,
                 'p3': None,
                 'birthplace3_rus': None,
                 'birthplace3_eng': None,
                 'birthplace3_check': False,
                 # part 2:
                 'p4a': None,
                 'p4b': None,
                 'p4ab_check': False,
                 'p4c_rus': None,
                 'p4c_eng': None,
                 'p4c_check': False,
                 'p5': None,
                 'p8_rus': None,
                 'p8_eng': None,
                 'p8_check': False,
                 'suggest': None  # may be Null
                 }

    def one_word(text, y, y2, what: str) -> bool:
        if abs(y - y2) < 17:
            if text:
                text = razd_sub_re.sub('', text)
                ret[what] = text
                return True
        return False

    def sentence(text, y1, y2, what: str) -> bool:
        if abs(y1 - y2) < 17:
            if text:  # != ''
                text = razd_sub_re.sub('', text)
                if ret[what] is None:
                    ret[what] = text
                else:
                    ret[what] += ' ' + text
                return True
        return False

    # Detect first line x and y
    # old
    # x_first = None
    # y_first = None
    # for i in range(len(ocr_rus['level'])):
    #     x = ocr_rus['left'][i]
    #     y = ocr_rus['top'][i]
    #     oe = ocr_rus['text'][i].upper()
    #     oe = re.sub(r'[\.\-),0-9]', '', oe)
    #     if len(oe) >= 3 and y < 70:  # search for first fam_rus
    #         x_first = x
    #         y_first = y
    #         break
    x_first = None
    y_first = None
    for i in range(len(ocr_rus['level'])):
        x = ocr_rus['left'][i]
        y = ocr_rus['top'][i]
        oe = ocr_rus['text'][i].upper()
        oe = re.sub(r'[^\.0-9]', '', oe)
        if oe != '' and find_date(oe) is not None:  # search for first date p 3
            x_first = x
            y_first = y - 115
            break

    # print(x_first, y_first)

    # top left corner
    # we declare fields by its offset
    step_m = 58
    if x_first is not None:
        x_names = x_first
    else:
        x_names = 55  # first x
    if y_first is not None:
        fam_rus_y = y_first
        # print("aaa", fam_rus_y, y_first)
    else:
        fam_rus_y = 25  # first y
    fam_eng_y = fam_rus_y + 25  # +25
    name_rus_y = fam_rus_y + 58  # 194  # +58
    name_eng_y = name_rus_y + 26
    y3 = name_rus_y + 57  # 252  # +58 +32
    birthplace3_rus_y = y3 + 29  # 281  # + 65
    birthplace3_eng_y = y3 + 59  # + 58
    y4 = birthplace3_rus_y + 65  # 342  # before cut # + 65 + 36
    # part 2
    p4b_x = x_names + 254
    p4c_rus_y = y4 + 33  # 375  # +34 y4 # ГИБДД
    p4c_eng_y = y4 + 58  # 401  # +58 y4 # GIBDD
    y5 = p4c_rus_y + 63  # +63 +38 # number # or 112
    place8_rus_y = y5 + 30
    place8_eng_y = y5 + 56

    # y5_coords = [0,0,0,0]
    # RUS better?
    # print(x_first, y_first)
    for i in range(len(ocr_rus['level'])):
        x = ocr_rus['left'][i]
        y = ocr_rus['top'][i]
        oe = ocr_rus['text'][i].upper()
        if abs(x - x_names) < 20 or 0 < (
                x - x_names) < 100:  # АЛЬ РИФАИ /mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/114-515-0.png
            if sentence(oe, y, fam_rus_y, 'fam_rus'):
                # fam_eng_y = y + 25
                # name_rus_y = y + 57
                continue
        if abs(x - x_names) < 20 or 0 < (x - x_names) < 300:
            if ret['name_rus'] is not None and y < name_rus_y:  # y может ошибочно подниматься у части номера
                name_rus_y = y
            if sentence(oe, y, name_rus_y, 'name_rus'):
                # y3 = y + 57  # used as y_first
                continue
        if abs(x - x_names) < 20 or 0 < (x - x_names) < 300:  # may be divided
            if sentence(oe, y, y3, 'p3'):  # 3 birth date
                birthplace3_rus_y = y + 29
                continue
        if abs(x - x_names) < 20 or 0 < (x - x_names) < 300:
            # y может ошибочно подниматься у части номера
            if ret['birthplace3_rus'] is not None and y < birthplace3_rus_y:
                birthplace3_rus_y = y
            if sentence(oe, y, birthplace3_rus_y, 'birthplace3_rus'):  # 3 birth place
                y4 = y + 65
                continue
        if abs(x - x_names) < 20:  # 4a)
            if one_word(oe, y, y4, 'p4a'):
                p4c_rus_y = y + 33
                y5 = y + 96
                p4b_x = x + 256  # x correction for second date
                continue
        if abs(x - p4b_x) < 40:  # 4b)
            if one_word(oe, y, y4, 'p4b'):
                p4c_rus_y = y + 33
                continue
        if abs(x - x_names) < 30 or 0 < (x - x_names) < 300:
            if sentence(oe, y, p4c_rus_y, 'p4c_rus'):  # 4c)
                y5 = y + 63
                continue
        if abs(x - x_names) < 20 or 0 < (x - x_names) < 300:
            if ret['p5'] is not None and y < y5:  # y может ошибочно подниматься у части номера
                y5 = y
            if sentence(oe, y, y5, 'p5'):
                # y5_coords[0] = x # fail
                # y5_coords[1] = y
                # y5_coords[2] = ocr_rus['height'][i]
                # y5 = 0  # disable
                place8_rus_y = y + 29
                place8_eng_y = y + 56
                continue
        if abs(x - x_names) < 20 or 0 < (x - x_names) < 300:
            if ret['p8_rus'] is not None and y < place8_rus_y:  # y может ошибочно подниматься у части номера
                place8_rus_y = y
            if sentence(oe, y, place8_rus_y, 'p8_rus'):  # 8)
                continue

    # b5 = binary[y5_coords[1] - 4: y5_coords[1]+y5_coords[2]+4, y5_coords[0]-4:250] # fail
    # print(pytesseract.image_to_string(b5, lang='rus', nice = 1))
    # print(ret['p5'])

    p3_rus = ret['p3']
    ret['p3'] = None
    # part2
    p4a_rus = ret['p4a']
    ret['p4a'] = None
    p4b_rus = ret['p4b']
    ret['p4b'] = None
    p5_rus = ret['p5']
    ret['p5'] = None

    # print(x_names, fam_eng_y)
    # ENG
    for i in range(len(ocr_eng['level'])):
        x = ocr_eng['left'][i]
        y = ocr_eng['top'][i]
        oe = ocr_eng['text'][i].upper()
        if abs(x - x_names) < 20 or 0 < (x - x_names) < 100:
            if sentence(oe, y, fam_eng_y, 'fam_eng'):
                # name_eng_y = y + step_m
                continue
        if abs(x - x_names) < 20 or 0 < (x - x_names) < 300:
            if ret['name_eng'] is not None and y < name_eng_y:  # y может ошибочно подниматься у части номера
                name_eng_y = y
            if sentence(oe, y, name_eng_y, 'name_eng'):
                # y3 = y + 31  # used as y_first
                continue
        if abs(x - x_names) < 20 or 0 < (x - x_names) < 300:  # may be divided
            if sentence(oe, y, y3, 'p3'):  # 3 birth date
                birthplace3_eng_y = y + 59
                y4 = y + 94
                continue
        if abs(x - x_names) < 20 or 0 < (x - x_names) < 300:
            # y может ошибочно подниматься у части номера
            if ret['birthplace3_eng'] is not None and y < birthplace3_eng_y:
                birthplace3_eng_y = y
            if sentence(oe, y, birthplace3_eng_y, 'birthplace3_eng'):  # 3 birth place
                # y4 = y + 35
                continue
        # part2
        if abs(x - x_names) < 20:  # 4a)
            if one_word(oe, y, y4, 'p4a'):
                p4c_eng_y = y + 59
                y5 = y + 96
                p4b_x = x + 256  # x correction for second date
                continue
        if abs(x - p4b_x) < 40:  # 4b)
            if one_word(oe, y, y4, 'p4b'):
                continue
        if abs(x - x_names) < 20 or 0 < (x - x_names) < 300:
            if sentence(oe, y, p4c_eng_y, 'p4c_eng'):  # 4c)
                y5 = y + 38
                continue
        if abs(x - x_names) < 20 or 0 < (x - x_names) < 300:
            if ret['p5'] is not None and y < y5:  # y может ошибочно подниматься у части номера
                y5 = y
            if sentence(oe, y, y5, 'p5'):  # 5)
                place8_eng_y = y + 57
                continue
        if abs(x - x_names) < 20 or 0 < (x - x_names) < 300:
            if ret['p8_eng'] is not None and y < place8_eng_y:  # y может ошибочно подниматься у части номера
                place8_eng_y = y
            if sentence(oe, y, place8_eng_y, 'p8_eng'):  # 8)
                continue

    p3_eng = ret['p3']
    ret['p3'] = compare_dates(p3_rus, p3_eng)
    # part2
    p4a_eng = ret['p4a']
    p4b_eng = ret['p4b']
    p5_eng = ret['p5']
    ret['p4a'] = compare_dates(p4a_rus, p4a_eng)
    ret['p4b'] = compare_dates(p4b_rus, p4b_eng)
    # serial number
    ret['p5'] = _compare_vod_snum(p5_rus, p5_eng)
    if ret['p5']:
        ret['p5'] = re.sub('[^0-9]', '', ret['p5'])

    def _ch(rus, lat, res):
        if ret[rus] and ret[lat]:
            if rus == 'birthplace3_rus':
                ret[res] = check(ret[rus], ret[lat], p3=True)
            else:
                ret[res] = check(ret[rus], ret[lat], p3=False)

    _ch('fam_rus', 'fam_eng', 'fam_check')
    _ch('name_rus', 'name_eng', 'name_check')
    _ch('birthplace3_rus', 'birthplace3_eng', 'birthplace3_check')
    _ch('p4c_rus', 'p4c_eng', 'p4c_check')
    _ch('p8_rus', 'p8_eng', 'p8_check')
    ret['p4ab_check'] = check_dates(ret['p4a'], ret['p4b'])

    # suggestion
    _suggest_dl(ret)  # side effect on ret

    return ret


def categories_rectangles(bin, gray) -> list or None:
    """ TODO: 1) detect rectangles even when character touch edge for [M] category
    TODO: 2) detect if we get two contours one in anather - internal and external

    :param bin:
    :param gray:
    :return: list of categories or None
    """

    # cv.imshow('image', bin)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q

    # select dilate power
    # binary = None
    cnts_save = []
    for i in range(3):
        bin2 = cv.dilate(bin, None, iterations=i)  # this or

        # bin2 = cv.erode(bin, None, iterations=i)
        cnts_raw, _ = cv.findContours(bin2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cnts = []
        for c in cnts_raw:
            area = cv.contourArea(c)
            (x, y, w, h) = cv.boundingRect(c)
            if 800 < area < 3100 and 0.4 < w / h < 3:  #
                # rectangle or not
                peri = cv.arcLength(c, True)
                approx = cv.approxPolyDP(c, 0.04 * peri, True)
                if len(approx) == 4:
                    cnts.append(c)
        if len(cnts) > len(cnts_save):
            cnts_save = cnts
    if cnts_save:
        cnts = cnts_save
    else:
        cnts = None

    # print('cnts', len(cnts))

    # process contours
    categories = set()
    if cnts is not None:

        for c in cnts:
            # cv.drawContours(image, [c], -1, (0, 0, 255), 2)
            rect = cv.minAreaRect(c)
            im = _crop_minAreaRect(gray, rect)

            # cv.imshow('image', im)  # show image in window
            # cv.waitKey(0)  # wait for any key indefinitely
            # cv.destroyAllWindows()  # close window q

            # print(im.shape)
            if im is None:
                logger.error("Error in prava recognition")
                continue
            cu = 5
            im = im[cu:im.shape[0] - cu, cu:im.shape[1] - cu]
            _, im = cv.threshold(im, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # to binary
            im = 255 - im  # required for cv.BORDER_CONSTANT
            b = 5  # add border
            im = cv.copyMakeBorder(im, b, b, b, b, cv.BORDER_CONSTANT)
            b = 0  #
            for _ in range(3):  # when we repeat symbol it start to recognize it
                im1 = cv.copyMakeBorder(im, b, b, b, b + im.shape[1], cv.BORDER_CONSTANT)
                im2 = cv.copyMakeBorder(im, b, b, b + im.shape[1], b, cv.BORDER_CONSTANT)
                im = cv.bitwise_or(im1, im2)
            im = 255 - im
            # im = cv.dilate(im, None, iterations=1)
            # cv.imshow('image', im)  # show image in window
            # cv.waitKey(0)  # wait for any key indefinitely
            # cv.destroyAllWindows()  # close window q
            c_string = '-c tessedit_char_whitelist=CD1EABmbM'
            with tesseract_image_to_string_lock:
                line = pytesseract.image_to_string(im, lang='ocrb', config=c_string)
                if not line:  # second try
                    im = cv.dilate(im, None, iterations=1)
                    line = pytesseract.image_to_string(im, lang='ocrb', config=c_string)

            line = re.sub('8', 'B', line.upper())  # TODO: add more
            # print(line)
            s = re.findall(r'([CD]1[E])|([ABCD][1mb])|([BCD]E)|([ABCDM])', line)
            if s:
                s = list(zip(*s))
                s1 = [x for x in s[0] if x != '']
                s2 = [x for x in s[1] if x != '']
                s3 = [x for x in s[2] if x != '']
                s4 = [x for x in s[3] if x != '']
                s1c = {x: s1.count(x) for x in s1}
                s2c = {x: s2.count(x) for x in s2}
                s3c = {x: s3.count(x) for x in s3}

                # print(s1, s2, s3, s4)
                # print(s1c, s2c, s3c)
                if s1 and s1c[s1[0]] > 2:
                    categories.add(s1[0])
                elif s2 and s2c[s2[0]] > 2:
                    categories.add(s2[0])
                elif s3 and s3c[s3[0]] > 2:
                    categories.add(s3[0])
                elif s4:  # if 1 symbol we can only trust most common
                    s4 = {x: s4.count(x) for x in s4}
                    if not s4:  # if empty
                        continue
                    mc = sorted(s4.values(), reverse=True)[0]
                    s4 = [key for key, value in s4.items() if value == mc][0]  # most common
                    categories.add(s4)
            # cv.drawContours(image, [c], -1, 255, 2)
    if categories:
        return list(categories)
    else:
        return None


def parse_back(binary) -> str or None:
    # cv.imshow('image', binary)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q
    t: dict = load('driving_license')
    try:
        structure = t["driving_license"][0]['back']['structure']
        x = structure[0]['position']['x']
        y = structure[0]['position']['y']
    except:  # any reason in load
        logger.exception('Template structure has failed')
        return None
    (height, width) = binary.shape
    x = int(x * width)
    y = int(y * height)  # 518 621 0.83 #0.79
    number_roi = binary[y:, x:]

    # cv.imshow('image', number_roi)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q
    # ---- SERIAL NUMBER parsing
    s_number = None
    try:
        with tesseract_image_to_string_lock:
            line = pytesseract.image_to_string(number_roi, lang='ocrb')
        # print(line)
        line = re_spaces.sub('', line).upper()
        # 1) without digitizing
        s_number = re.sub('[^0-9]', '', line)
        if len(s_number) > 10:
            s_number = s_number[-10:]  # only last smbols

        if len(s_number) != 10:
            # 2) digitizing
            s_number = letters_to_digits(line)  # digitize letters
            s_number = re.sub('[^0-9]', '', s_number)
            if len(s_number) > 10:
                s_number = s_number[-10:]  # only last smbols
        # print(line)
    except TesseractError as ex:
        logger.error("ERROR: %s" % ex.message)

    if s_number and len(s_number) == 10:
        return s_number
    else:
        return None


def crop_and_parse(image, width, m_counts):
    # read: width, m_counts

    img_cr, cr_index = pcr.crop(image, m_counts)
    ret_obj = Anonymous()

    if img_cr is None:
        return Anonymous("Driving license not recognized.", 4)

    gray = cv.cvtColor(img_cr, cv.COLOR_BGR2GRAY)
    gray = imutils.resize(gray, width=width)  # resized

    if gray is None:
        return Anonymous("Error in driving license parser.", 4)

    # FRONT SIDE
    if cr_index <= 1:  # 0, 1
        ret_obj.OUTPUT_OBJ['side'] = 'front'

        r, _ = threashold_by_letters(gray, amount=160, scale_factor=2)

        if r is None:
            return Anonymous("Fail to detect front side of Driving License.", 4)

        r = r[100:, 350:]  # delete left part
        gray = gray[100:, 350:]  # delete left part

        main_text = {}
        mt = parse_front_text(r)

        if mt is not None:
            main_text.update(mt)

        # CATEGORIES recognition
        r = r[(r.shape[0] - 200):, :]  # 626
        gray = gray[(gray.shape[0] - 200):, :]
        main_text['categories'] = categories_rectangles(r, gray)
        check_status = 0
        if mt is not None:
            check_status = sum((mt['fam_check'] is True, mt['fam_check'] is True, mt['name_check'] is True,
                                mt['birthplace3_check'] is True, mt['p4c_check'] is True,
                                mt['p8_check'] is True))
        if mt is not None and check_status == 6 and main_text['categories'] is not None:
            ret_obj.OUTPUT_OBJ['qc'] = 0
        elif mt is not None and main_text['categories'] is not None:
            ret_obj.OUTPUT_OBJ['qc'] = 1
        elif mt is not None:
            ret_obj.OUTPUT_OBJ['qc'] = 2

        ret_obj.OUTPUT_OBJ.update(main_text)
    # BACK SIDE
    elif cr_index >= 2:
        ret_obj.OUTPUT_OBJ['side'] = 'back'

        gray = fix_angle(gray, get_lines_c)
        r, _ = threashold_by_letters(gray, amount=40, scale_factor=2)

        if r is None:
            return Anonymous("Fail to detect back side of Driving License.", 4)

        # BACK PARSER
        s_number = parse_back(r)
        if s_number is None:
            return Anonymous("Fail to detect back side of Driving License.", 4)

        ret_obj.OUTPUT_OBJ['qc'] = 1

        ret_obj.OUTPUT_OBJ['s_number'] = s_number

    # print(anonymous_return.OUTPUT_OBJ)
    else:
        ret_obj.OUTPUT_OBJ['qc'] = 4
    return ret_obj


def resize_for_akaze(image):
    if image.shape[0] > image.shape[1]:  # most common 2479x3508
        sh = image.shape[0]  # h
        if sh > 5000:
            image = imutils.resize(image, width=2500)
    else:
        sh = image.shape[0]  # w
        if sh > 5000:
            image = imutils.resize(image, height=2500)  # no metter width or
    return image


def dl_parse(obj: DocTypeDetected):
    """  prava 1)front new 2) front old 3) back old

    :param obj: with original image
    :return: class with OUTPUT_OBJ
    """

    image = obj.original

    try:
        # 1) RESIZE FOR AKAZE
        height = 674  # average prep
        width = 998  # average
        # resize too big images
        image = resize_for_akaze(image)

        # 2) AKAZA cropper
        m_counts: list = pcr.match(image)  # match to kaze_templates
        OutputToRedis.progress(obj, 55)

        if m_counts is None or sum(m_counts) == 0:
            return Anonymous("Fail to match driver license to templates.", 4)

        # 3) main work
        ret = crop_and_parse(image, width, m_counts)
        OutputToRedis.progress(obj, 80)
        # two retries
        if ret.OUTPUT_OBJ['qc'] >= 2:
            OutputToRedis.progress(obj, 80)
            Anonymous.OUTPUT_OBJ = dict(qc=3, side=None)  # clear
            ret = crop_and_parse(image, width, m_counts)
            if ret.OUTPUT_OBJ['qc'] >= 2:
                OutputToRedis.progress(obj, 90)
                Anonymous.OUTPUT_OBJ = dict(qc=3, side=None)  # clear
                ret = crop_and_parse(image, width, m_counts)

        return ret

    except Exception as e:
        logger.exception("Uncatched exception in Driving license")
        if hasattr(e, 'message'):
            return Anonymous("Driving license: " + e.message, 4)
        else:
            return Anonymous("Driving license: " + str(type(e).__name__) + " : " + str(e.args), 4)

    #         cv.imshow('image', im)  # show image in window
    #         cv.waitKey(0)  # wait for any key indefinitely
    #         cv.destroyAllWindows()  # close window q

    # r = np.zeros_like(r) + 100
    # print(r.shape)
    # r = (r + 100) /2
    # cv.drawContours(image, newc, -1, 255, 2)
    # # cv.drawContours(image, newc, -1, 255, 2)
    # cv.imshow('image', image)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q

    # parse_main_text(r)

    # def IOU(bbox1, bbox2) -> float:


#     """ Calculate overlap between two bounding boxes [x, y, w, h] as the area of intersection over the area of unity
#
#     :param bbox1:
#     :param bbox2:
#     :return: percent of overlap, 0 - no overlap
#     """
#     x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
#     x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
#
#     w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
#     h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
#     if w_I <= 0 or h_I <= 0:  # no overlap
#         return 0
#     I = w_I * h_I
#     U = w1 * h1 + w2 * h2 - I
#     return I / U
#
#
# import numpy as np
# import scipy.signal

# def cross_image(im1, im2):
#    # get rid of the color channels by performing a grayscale transform
#    # the type cast into 'float' is to avoid overflows
#    im1_gray = np.sum(im1.astype('float'), axis=2)
#    im2_gray = np.sum(im2.astype('float'), axis=2)
#
#    # get rid of the averages, otherwise the results are not good
#    im1_gray -= np.mean(im1_gray)
#    im2_gray -= np.mean(im2_gray)
#
#    # calculate the correlation image; note the flipping of onw of the images
#    return scipy.signal.fftconvolve(im1_gray, im2_gray[::-1, ::-1], mode='same')


if __name__ == '__main__':
    # img1, gray = crop(img1, rotate=False, rate=0.22)

    # old
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/72-204-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/0/8-139-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/85-486-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/85-383-0.png'  # orig
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/84-382-0.png'  # crivaja
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/64-362-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/208-609-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/42-340-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/45-287-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/184-585-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/2/6-247-11.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/2/47-178-9.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/4-302-0.png'
    # obr old
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/275-676-1.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/66-467-1.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/91-223-1.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/34-276-1.png'
    p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/168-569-1.png'  # TODO
    # new
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/207-608-0.png'  # new
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/70-368-0.png'  # new
    p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/4-245-0.png'
    # obr new
    p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/15-313-1.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/1/38-48-10.png'
    # strange rus
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/2/56-354-11.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/1/60-358-0.png'
    import UtilModule
    import os

    os.mkdir('1')
    p = '/mnt/hit4/hit4user/Downloads/ДКП,счет,фото,взнос,птс.pdf'
    filelist = UtilModule.UtilClass.PdfToPng(p, '1')  # split
    exit(0)
    # img = cv.imread(p)
    # bgr_not_resized_cropped, gray = crop(img, rotate=False, rate=0.26)
    # cv.imshow('image', bgr_not_resized_cropped)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q
    # exit(0)
    # read_rectangles_template()
    # obr_old_v = '/home/u2/Desktop/obr_old_v4_test.png'
    # image = cv.imread(obr_old_v)
    # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # # ret, image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
    # cv.imshow('image', image)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q
    # z = np.zeros((image.shape[0],image.shape[1],3), np.uint8)
    # # ret, thresh = cv.threshold(image, 127, 255, 0)
    # (_, cnts, _) = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #
    # # (_, cnts, _) = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # for c in cnts:
    #     x, y, w, h = cv.boundingRect(c)
    #     cv.rectangle(z, (x, y), (x + w, y + h), (0, 255, 0), 1)
    #
    # cv.imshow('image', z)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q
    #
    # o = '/home/u2/Desktop/new_obr_v3.png'
    #
    # image = cv.imread(o, cv.IMREAD_COLOR)
    # # image = cv.blur(image, (2, 2))
    # image = cv.fastNlMeansDenoising(image, h=8, templateWindowSize=10)  # denoise edges
    # cropper1 = KazeCropper(image)
    # image = cv.imread(p, cv.IMREAD_COLOR)
    # # print(cropper1.match(image))
    # image = cropper1.crop(image)
    # print(image)

    # from matplotlib import pyplot as plt
    # plt.imshow(image)
    # plt.show()
    # exit(0)

    # DIRECTORY
    # p = '/home/u2/Desktop/passport_recognition_example2/'
    # import os
    #
    # for i, filename in enumerate(os.listdir(p)):
    #     if i < 3:
    #         continue
    #     image = cv.imread(p + filename, cv.IMREAD_COLOR)  # scanned image
    #     print(i, p + filename)
    #     o = dl_parse(image)
    #     print(o.OUTPUT_OBJ)
    # exit(0)

    # SINGLE FILE
    p = '/home/u2/a.jpg'
    image = cv.imread(p, cv.IMREAD_COLOR)  # scanned image
    o = dl_parse(image)
    print(o.OUTPUT_OBJ)

    # bgr_not_resized_cropped, gray = crop(image, rotate=True, rate=0.22)
    # image = bgr_not_resized_cropped
    # image, cr_index = cr.crop(image)
    # # KAZE -------------- end
    # if image is None:
    #     print("BAD1")
    #     exit(1)

    # image = fix_angle(image, get_lines_c)
    # from matplotlib import pyplot as plt
    #
    # plt.imshow(image)
    # plt.show()
    #
    # # cv.imwrite('tmp5.png', img)
    #
    # #
    # # PREPARE --------- start
    # # img = cv.imread('tmp2.png', cv.IMREAD_COLOR)
    # # img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #
    # # cv.imwrite('tmp5.png', img)
    # #
    # #
    #
    # # print(chans[1].dtype)
    # # chans[1] = np.multiply(chans[1], 0.2, casting='unsafe').astype(int)  #    chans[1] *= (chans[1] * 0.2).
    # p = '/home/u2/Desktop/t.png'
    # image = cv.imread(p, cv.IMREAD_COLOR)  # scanned image
    # # image = fix_angle(image, get_lines_c)
    #
    # # cv.imshow('image', image)  # show image in window
    # # cv.waitKey(0)  # wait for any key indefinitely
    # # cv.destroyAllWindows()  # close window q
    # from matplotlib import pyplot as plt
    #
    # plt.imshow(image)
    # plt.show()
    # cv.imwrite('tmp6.png', image)
    # exit(0)
    #
    # dl_parse(image)

    # PREPARE --------- end

    # grab the image channels, initialize the tuple of colors,
    # the figure and the flattened feature vector
    # chans = cv.split(img)

    # colors = ("b", "g", "r")
    # from matplotlib import pyplot as plt
    # plt.figure()
    # plt.title("'Flattened' Color Histogram")
    # plt.xlabel("Bins")
    # plt.ylabel("# of Pixels")
    # features = []
    # for (chan, color) in zip(chans, colors):
    #     # create a histogram for the current channel and
    #     # concatenate the resulting histograms for each
    #     # channel
    #     hist = cv.calcHist([chan], [0], None, [256], [0, 256])
    #     features.extend(hist)
    #
    #     # plot the histogram
    #     plt.plot(hist, color=color)
    #     plt.xlim([0, 256])
    #

    # # define the list of boundaries
    # boundaries = [
    #     ([17, 15, 100], [50, 56, 200]),
    #     ([86, 31, 4], [220, 88, 50]),
    #     ([25, 146, 190], [62, 174, 250]),
    #     ([103, 86, 65], [145, 133, 128])
    # ]

    # # loop over the boundaries
    # for (lower, upper) in boundaries:
    #     # create NumPy arrays from the boundaries
    #     lower = np.array(lower, dtype="uint8")
    #     upper = np.array(upper, dtype="uint8")
    #
    #     # find the colors within the specified boundaries and apply
    #     # the mask
    #     mask = cv.inRange(img, lower, upper)
    #     output = cv2.bitwise_and(img, img, mask=mask)
    #
    #     # show the images
    #     cv2.imshow("images", np.hstack([image, output]))
    #     cv2.waitKey(0)

    # lower = np.array([120, 57, 110])  # -- Lower range -- RGB
    # upper = np.array([180, 136, 170])  # -- Upper range --
    # mask = cv.inRange(img, lower, upper)
    # res = cv.bitwise_and(img, img, mask=mask)
    # chans[0] = chans[0]*0
    # print(chans[0].shape)
    # # chans[2] *= 0
    # # chans[0] *= 0
    # chans[1] *= 0
    # img = cv.merge(chans)
    #
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # ret2, r = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # cv.imshow("images", r)
    # cv.waitKey(0)

    # im = cv.merge(chans)
    # plt.imshow(im)
    # plt.show()
    # from matplotlib import pyplot as plt
    # plt.imshow(img)
    # plt.show()
    # cv.imshow('image', res)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q
    # exit(0)
    #
