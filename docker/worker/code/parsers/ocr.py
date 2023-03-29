import pytesseract
# from pytesseract import Output
from pytesseract.pytesseract import TesseractError
import numpy as np
import cv2 as cv
#own
from logger import logger
from tesseract_lock import tesseract_image_to_string_lock, tesseract_image_to_boxes_lock


rus_alph = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
eng_alph = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
punct = '.-—'
numbers = '0123456789'
rus_bad = r'!\"#$%&\'()*+,/:;<=>?@[\]^_`{|}©«®°»і‘’‚“”„…‹›№'  # without абвгдежзийклмнопрстуфхцчшщъыьэюяё
eng_bad = r'!\"#$%&\'()*+,/:;<=>?@[\]_{|}~¢£¥§©«®°»é‘’“”€ﬁﬂ'  # without abcdefghijklmnopqrstuvwxyz

# https://habr.com/en/company/recognitor/blog/225913/

# rus_all = rus_alph + ' ' + numbers
# rus_all = rus_alph
# eng_all = eng_alph + ' ' + numbers
# eng_all = eng_alph


def image_to_string(image, lang: str, white_list: str = None, blacklist: str = None) -> str or None:
    """
    It working ban on the multithreading.

    :param image: prepared
    :param lang:
    :param white_list:
    :return:
    """
    with tesseract_image_to_string_lock:
        text = None
        try:
            if white_list:
                text = pytesseract.image_to_string(image, lang=lang, config='-c tessedit_char_whitelist=' + white_list)
            elif blacklist:
                text = pytesseract.image_to_string(image, lang=lang, config='-c tessedit_char_blacklist=' + blacklist)
            else:
                text = pytesseract.image_to_string(image, lang=lang)
        except TesseractError as ex:
            logger.info("cannot recognize text: %s" % ex.message)
        return text


def roi_preperation(img, i, scale_factor: int = 1) -> np.ndarray:
    """ Image preperation for pytesseract OCR

    :param img: gray
    :param i: one of range(10)
    :param scale_factor: must be 1 or 2 for passport and prava
    :return: image binary
    """
    if scale_factor == 1:  # passport
        denoising = 2 + 2 * i
        templateWindowSize = 10
    elif scale_factor == 2:  # prava
        denoising = 1 + i // 2
        templateWindowSize = 4
    else:
        logger.critical("parsers.ocr: no scale_factor!")
    r = img
    r = 255 - r
    if i != 0:
        r = cv.fastNlMeansDenoising(r, h=denoising,
                                    templateWindowSize=templateWindowSize)  # h=40, templateWindowSize=20)
    if i % 3 == 0:
        r = cv.dilate(r, None, iterations=1)  # расширить?
    if i % 2 == 0:
        r = cv.erode(r, None, iterations=1)  # разъесть?
    # kernel = np.ones((2, 2), np.uint8)  # left

    ret2, r = cv.threshold(r, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    contours, _ = cv.findContours(r, cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)
    # print(contours)
    # remove the bad contours
    # initialize the mask that will be used
    mask = np.ones(r.shape[:2], dtype="uint8") * 255
    for i in range(len(contours)):
        # print(contours[i])
        a = cv.contourArea(contours[i])
        # print(a)
        if a < 8:  # too small contours will be erased
            # print(a)
            # cv.drawContours(mask, [contours[i]], -1, 0, -1)
            r = cv.bitwise_and(r, r, mask=mask)
    # th = cv.erode(th, kernel, iterations=1)

    r = 255 - r
    return r


def digits_to_rus_letters(s: str) -> str:
    """ used by passport_bottom_kaze

    :param s:
    :return: str
    """
    s = s.replace('0', 'О')
    s = s.replace('3', 'З')
    s = s.replace('5', 'Б')
    s = s.replace('8', 'В')
    return s


def threashold_by_letters(gray, amount=160, scale_factor=2) -> np.ndarray or None:
    """
    :param scale_factor: must be 1 or 2 for passport and prava
    :return: np.ndarray or None if letters_count < amount / 2
    """
    r_ret = None
    r_max = 0
    r_let = None
    for i in range(10):  # recognition loop
        try:
            r = roi_preperation(gray, i, scale_factor=scale_factor)
            with tesseract_image_to_boxes_lock:
                letters = pytesseract.image_to_boxes(r, lang='rus')
            letters = letters.split('\n')
            letters = [letter.split() for letter in letters]
            lc = len(letters)
            # print(lc)
            # >160 or max saved
            if lc > amount:
                r_ret = r
                r_max = lc
                r_let = letters
                break
            elif lc > r_max:
                r_ret = r
                r_max = lc
                r_let = letters
        except TesseractError as ex:
            logger.error("ERROR: %s" % ex.message)
    if r_max < amount / 2:
        return None, None
    if r_let:
        r_let = ''.join([x[0] for x in r_let if x])
    return r_ret, r_let

# pytesseract.pytesseract.tesseract_cmd = r"/home/u/.local/bin/pytesseract"

# try:
#     # tessdata_dir_config = '--tessdata-dir "/mnt/hit4/hit4user/PycharmProjects/rec2/"'
#     # pytesseract.pytesseract.tesseract_cmd = r"pytesseract"
#     if lang == 'ocrb':
#         # Example config: '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
#         # It's important to include double quotes around the dir path.
#
#         # pytesseract.image_to_string(image, lang='chi_sim', config=tessdata_dir_config)
#         # text = pytesseract.image_to_string(r, lang='eng')
#
#         text = pytesseract.image_to_string(r, lang='ocrb')
#     elif lang == 'rus':
#         text = pytesseract.image_to_string(r, lang='rus')
#     # elif lang == 'digits':
#     #    text = pytesseract.image_to_string(r, lang='digits', config=tessdata_dir_config) #, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
#     else:
#         text = pytesseract.image_to_string(r, lang='eng')
# except TesseractError as ex:
# sys.stderr.write("ERROR: %s" % ex.message)
# # sys.exit(ex.status)