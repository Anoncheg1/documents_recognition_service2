import re
import cv2 as cv
import numpy as np
from concurrent.futures import ThreadPoolExecutor
# own
from parsers.passport_utils import roi_ocr, sort_contours_lines, enlarge_boxes, find_date, razd
from parsers.passport_utils import search_with_error
from parsers import ocr
from parsers.passport_bottom_kaze import ParserBottomKaze
from utils.progress_and_output import OutputToRedis, PROFILING
# from utils.progress_and_output import OutputToRedis

bottom_parser = ParserBottomKaze


def page_sep(img, i: int) -> (int, int, int, int):
    """ to gray and get contours
    not working for scans with many black dots"""
    # a = cv.split(img)
    # # a[1] = a[1] + 10
    # # a[1] = a[1] - 10
    # # a[2] = a[2] - 10
    # img = cv.merge(a)
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = img
    if i > 1:
        gray = cv.fastNlMeansDenoising(gray, h=(10 * i), templateWindowSize=20)  # h=40, templateWindowSize=20)

    # initialize a rectangular and square structuring kernel
    rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (60, 4))
    # sqKernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))

    gray = cv.GaussianBlur(gray, (3, 3), 0)
    # gray
    gray = cv.Sobel(gray, ddepth=cv.CV_32F, dx=0, dy=1, ksize=-1)
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    # compute the Scharr gradient of the blackhat image and scale the
    # result into the range [0, 255]
    gradX = cv.Sobel(gray, ddepth=cv.CV_32F, dx=0, dy=1, ksize=-1)
    # gradX = cv.Sobel(gradX, ddepth=cv.CV_32F, dx=0, dy=1, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
    # apply a closing operation using the rectangular kernel to close
    # gaps in between letters -- then apply Otsu's thresholding method
    thresh = gradX
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, rectKernel)

    thresh = cv.threshold(thresh, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    thresh = cv.dilate(thresh, None, iterations=1)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, rectKernel)

    thresh = cv.erode(thresh, None, iterations=4)

    # during thresholding, it's possible that border pixels were
    # included in the thresholding, so let's set 30% of borders to zero
    # print(img.shape)
    hp = int(img.shape[0] * 0.25)
    hp2 = img.shape[0] - hp
    wp = int(img.shape[1] * 0.13)
    wp2 = img.shape[1] - wp

    thresh[:, 0:wp] = 0  # x
    thresh[0:hp, :] = 0  # y
    thresh[:, wp2:] = 0  # img.shape[1] width
    thresh[hp2:, :] = 0  # img.shape[0] height

    cnts, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)
    res = None
    gap = 40
    cc = None
    # print(wp)
    # print(wp2)
    for c in cnts:
        box = cv.boundingRect(c)
        x, y, w, h = box
        # print(x, y, x+ w, h)
        if wp - gap < x < wp + gap \
                and wp2 - gap < (x + w) < wp2 + gap \
                and 130 > h > 10:
            res = box
            cc = c
            break
    # if cc is not None:
    #     cv.drawContours(img, [cc], -1, (0, 0, 255), 5)
    # cv.drawContours(img, cnts, -1, (0, 255, 0), 3)

    # # if res is None:
    # #     res = 0, img.shape[0]//2, 0, 0
    # print(res)
    # img2 = cv.resize(img, (900, 900))
    # # print(res)
    # cv.imshow('image', img2)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q
    return res


def get_gender(line: str):
    gender = None
    res = re.search(r'([^А-Я]|^| )([' + ocr.rus_alph + r']{2,3})([^А-Я]|$| )', line)
    if res:
        sp = res.span(2)  # only gender
        tmp = line[sp[0]:sp[1]]
        if search_with_error(tmp, 'МУЖ'):
            gender = 'МУЖ'
        elif search_with_error(tmp, 'ЖЕН'):
            gender = 'ЖЕН'
    return gender


def main_bottom_page(lines: list, gray) -> (str, str, str):
    """
    ФИО определяется только состоящее из 1 слова если оно более 2 символов.
    Читаются слова сверху вниз слева направо.
    """
    ret = {'F': None,
           'I': None,
           'O': None,
           'gender': None,
           'birth_date': None,
           'birth_place': '',
           }

    for l_ix, b_line in enumerate(lines):
        line = ''
        for w_ix, b in enumerate(b_line):  # word in line
            x, y, w, h = b
            roi = gray[y:y + h, x:x + w].copy()
            text = roi_ocr([roi], lang='rus', blacklist=ocr.rus_bad)  # get text in regions
            if text:
                text = text[0]
            # print(text)
            if text:
                t1 = ' '.join(text).upper()

                if t1 != '':
                    line += t1 + ' '
                    if len(t1) >= 3:
                        if not ret['birth_date'] and not ret['gender']:  # must be at one line
                            if not ret['birth_date']:
                                ret['birth_date'] = find_date(line)  # require line
                            if not ret['gender']:
                                ret['gender'] = get_gender(line)  # part of line

                        if ret['gender'] or ret['birth_date']:  # at same line of after
                            for t_part in text:
                                t_part = t_part.upper()
                                # print(t_part)
                                spaces = sum(1 for c in t_part if c == ' ')
                                length = len(t_part) - spaces
                                if length > 0 and sum(1 for c in t_part if
                                                      c != ' ' and c.isupper()) / length > 0.7 \
                                        or t_part.upper() == 'ГОР':  # most symbols are upper TODO: remake for upper
                                    if get_gender(t_part) is None:  # exclude gender
                                        ret['birth_place'] += t_part + ' '

                        re_fio = re.search(r'[' + ocr.rus_alph + ']{3,}', t1)

                        if re_fio:  # FIO and gender
                            sp = re_fio.span()
                            t = t1[sp[0]:sp[1]]
                            if not ret['birth_date'] and not ret['gender'] \
                                    and not re.search(r'ФАМИЛИЯ', t1) and t != 'ИМЯ' \
                                    and not re.search(r'ОТЧЕСТВО', t1) and t != 'ПОЛ' \
                                    and t != 'МУЖ' and t != 'ЖЕН':
                                if ret['F'] and not ret['I'] and 0 < l_ix < 7:
                                    ret['I'] = t
                                elif ret['F'] and not ret['O'] and 1 < l_ix < 8:
                                    ret['O'] = t

            # cv.rectangle(gray, (x, y), (x + w, y + h), (255, 255, 255), 2)  # Debug!
        # print(line)
        # per line
        if line != '':
            if not ret['birth_date'] and not ret['gender']:  # at one line
                if not ret['F'] and l_ix < 4:  # multiline
                    ret['F'] = line.strip()
                ret['birth_date'] = find_date(line)
                if not ret['gender']:
                    ret['gender'] = get_gender(line)
                    # print(line)
                    res = re.search(r'([^А-Я]|^| )([' + ocr.rus_alph + r']{2,3})([^А-Я]|$| )', line)
                    if res:
                        sp = res.span(2)  # only gender
                        tmp = line[sp[0]:sp[1]]
                        if search_with_error(tmp, 'МУЖ'):
                            ret['gender'] = 'МУЖ'
                        elif search_with_error(tmp, 'ЖЕН'):
                            ret['gender'] = 'ЖЕН'
                # if gender or birth_date:
                #     continue

    # print('FIO:', ret['F'], '|', ret['I'], '|', ret['O'])
    # print('birth_date, gender, m_rod:', ret['birth_date'], '|', ret['gender'], '|', ret['birth_place'])

    # img2 = cv.resize(gray, (900, 900))
    # cv.imshow('image', img2)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q
    if ret['birth_place'] == '':
        ret['birth_place'] = None
    return ret


def main_top_page(lines: list, hs, ws, middle_h, gray):
    vidan = ''
    data_vid = None
    code_pod = None

    for i_line, b_line in enumerate(lines):  # line

        line = ''
        for i_word, b in enumerate(b_line):  # word in line
            x, y, w, h = b
            roi = gray[y:y + h, x:x + w].copy()
            ret = roi_ocr([roi], lang='rus', blacklist=ocr.rus_bad)
            text = ret[0]
            if text:
                line += ' '.join(text).upper() + ' '
                for t_part in text:
                    t_part = t_part.upper()
                    if t_part != '' and not code_pod and not data_vid and i_line > 0:  # not first line
                        spaces = sum(1 for c in t_part if c == ' ')
                        length = len(t_part) - spaces
                        # most symbols are upper TODO: remake for upper
                        if length > 0 and sum(1 for c in t_part if c != ' ' and c.isupper()) / length > 0.7:
                            vidan += t_part + ' '

            # cv.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Debug!

        # print(i_line, line)
        if not data_vid and line != '':
            data_vid = find_date(line)
        if not code_pod and line != '':
            res = re.search(r'([^0-9]|^)([0-9]{3}' + razd + r'[0-9]{3})([^0-9]|$)', line)
            if res:
                sp = res.span(2)  # only gender
                code_pod = line[sp[0]:sp[1]]

    vidan = vidan.strip()
    if vidan == '':
        vidan = None
    # print(vidan, '|', data_vid, '|', code_pod)
    return_value = {'vidan': vidan,
                    'data_vid': data_vid,
                    'code_pod': code_pod
                    }
    return return_value

    # img2 = cv.resize(gray, (900, 900))
    #
    # cv.imshow('image', img2)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window


def photo_box(thresh, y2_middl):
    """
    :param thresh:
    :param y2_middl:
    :return: box
    """
    height, width = thresh.shape
    h_min = height / (4.9 + 1)
    h_max = height / (4.9 - 1)
    w_min = width / (3.5 + 0.7)
    w_max = width / (3.5 - 0.7)

    # print(thresh.shape)
    # print(h_min, h_max, w_min, w_max)

    cross = cv.getStructuringElement(cv.MORPH_CROSS, (27, 27))
    x = cv.morphologyEx(thresh, cv.MORPH_OPEN, cross, iterations=1)

    cnts, _ = cv.findContours(x, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # mask = np.zeros(thresh.shape[:2], dtype="uint8")  # * 255
    box = None
    for contour in cnts:
        box = cv.boundingRect(contour)
        x, y, w, h = box
        if y > y2_middl and h_min < h < h_max and w_min < w < w_max:
            return box
            # cv.rectangle(mask, (x, y), (x + w, y + h), 255, -1)  # Debug!

    # cv.drawContours(mask, cnts, -1, 255, 1)
    # print(thresh.shape)
    # img2 = cv.resize(mask, (900, 900))
    # import matplotlib.pyplot as plt
    # plt.imshow(img2)
    # plt.show()

    # cv.imshow('image', img2)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q

    return box


def main_two_pages(cnts, gray, thresh, mrz_y_top, obj) -> (str, str, str):
    """

    :param cnts:
    :param gray:
    :param thresh:
    :param mrz_y_top:
    :param obj: for progress output
    :return:
    """
    h_img, w_img = gray.shape
    # print(h_img, w_img)

    hs = h_img // 180
    ws = w_img // 180  # min limits
    hh = h_img // 8  # hight limit
    # wh = w_img // 2.6

    # middle_h = h_img // 2

    # muddle
    res = None
    # y1_middl, y2_middl = None, None
    for i in range(9):
        res = page_sep(gray, 9 - i)
        if res is not None:
            break
    if res is not None:
        if res[3] < 100:
            y1_middl = res[1]
            y2_middl = res[1] + 100
        else:
            # print(res[0])
            y1_middl = res[1]
            y2_middl = res[1] + res[3]
    else:
        y1_middl = h_img / 2
        y2_middl = w_img / 2

    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # ret2, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # image box detection
    box = photo_box(thresh, y2_middl)
    if box is not None:
        w_photo = box[0] + box[2] + int(box[2] / 6.8)
    else:
        w_photo = w_img * 0.82
    # print(w_img * 0.82)
    # print(box[0] + box[2] + int(box[2] / 6.8))

    # import matplotlib.pyplot as plt
    # plt.imshow(gray)
    # plt.show()

    # text boxes filter
    boxes_filtered = []
    for contour in cnts:
        x, y, w, h = cv.boundingRect(contour)
        # Don't plot small false positives that aren't text
        if w < ws or h < hs or h > hh:  # or w > wh:  # or h < 10 or w > 900:
            continue
        if mrz_y_top and y < mrz_y_top:  # skip cnts which lower than mrz
            continue
        boxes_filtered.append((x, y, w, h))

    # filter for bottom page
    bottom_boxes = []
    for box in boxes_filtered:
        x, y, w, h = box
        if y < y2_middl \
                or x < w_photo or x > (w_img * 0.82):
            # or x < (w_img * 0.28) or x > w_photo:  # 28% left 18% rights

            #
            # 28% left 18% rights
            continue
        bottom_boxes.append((x, y, w, h))

    # filter for top page
    top_boxes = []
    for box in boxes_filtered:
        x, y, w, h = box
        if y > y1_middl \
                or x < (w_img * 0.12) or x > w_img * 0.82:  # 13% left 18% rights
            continue
        top_boxes.append((x, y, w, h))

    bottom_boxes = enlarge_boxes(bottom_boxes, ws, hs)
    top_boxes = enlarge_boxes(top_boxes, ws, hs)

    # for b in bottom_boxes:
    #     x, y, w, h = b
    #     cv.rectangle(gray, (x, y), (x + w, y + h), 255, -1)  # Debug!
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(gray)
    # plt.show()

    bottom_lines = sort_contours_lines(bottom_boxes, ws, hs)
    top_lines = sort_contours_lines(top_boxes, ws, hs)
    if PROFILING:
        # 1)
        mtp = main_top_page(top_lines, hs, ws, y1_middl, gray)  # 'main_top_page' STRUCTURE inside!
        OutputToRedis.progress(obj, 60)
        # 2) New kaze approach first for bottom !
        mbp1 = bottom_parser.parse(gray)  # gray - cropped

        OutputToRedis.progress(obj, 80)
        # 3)  Old approach without kaze:
        mbp2 = main_bottom_page(bottom_lines, gray)  # 'main_bottom_page' STRUCTURE inside!
    else:
        with ThreadPoolExecutor(max_workers=3) as executor:
            top = executor.submit(main_top_page, top_lines, hs, ws, y1_middl, gray)
            bottom_kaze = executor.submit(bottom_parser.parse, gray)
            bottom_old = executor.submit(main_bottom_page, bottom_lines, gray)
            mtp = top.result()
            mbp1 = bottom_kaze.result()
            mbp2 = bottom_old.result()
    # print("mtp", mtp)
    # print("mbp", mbp1)
    # print("mbp2", mbp2)

    for k,v in mbp1.items():
        if v is None and mbp2[k]:
            mbp1[k] = mbp2[k]
    # mbp=mbp2

    return mtp, mbp1
