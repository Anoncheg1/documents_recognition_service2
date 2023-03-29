import re
import numpy as np
import cv2 as cv
# own
from parsers.passport_utils import sort_contours_lines, search_with_error

#t =['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<']

chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
rus = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
rus_codes ='ABVGDE2JZIQKLMNOPRSTUFHC34WXY9678'
# Digits to letters 0158 0-O 1-I, 5-S
# letters to digits OISB-0158

re_spaces = re.compile(r' +')

dict_codes_rus = dict(zip(rus_codes, rus))


def digits_to_letters(s: str) -> str:
    """ Good for OCRB

    :param s:
    :return: str
    """
    s = s.replace('0', 'O')
    s = s.replace('1', 'I')
    s = s.replace('5', 'S')
    # s = s.replace('8', 'B') 8 - Я
    return s


def letters_to_digits(s: str) -> str:
    """ Good for OCRB
    and remove not digits

    :param s:
    :return: str
    """
    s = s.replace('O', '0')
    s = s.replace('I', '1')
    s = s.replace('S', '5')
    s = s.replace('B', '8')
    s = s.replace('<', '0')
    s = re.sub(r"\D", "", s)
    return s


def search_fio_separator(source: str) -> tuple or None:
    """

    :param source:
    :param sub:
    :return: (start,end) or None
    """
    source = source.upper()

    f = re.search('<<', source)  # what we search
    if f:
        return f.span()

    f = re.search('K<', source)
    if f:
        return f.span()

    f = re.search('<K', source)
    if f:
        return f.span()

    f = re.search('<e', source)
    if f:
        return f.span()

    f = re.search('e<', source)
    if f:
        return f.span()

    f = re.search('X<', source)
    if f:
        return f.span()

    f = re.search('<X', source)
    if f:
        return f.span()

    f = re.search('<', source)  # if nothing else
    if f:
        return f.span()

    return None


def mrz_control(mrz_part: str) -> (str or None, bool or None):
    """ Check control sum. For 1-9, 14-19, 29-42
    :param mrz_part:
    :return: serial ; True - no error, False - has error
    """
    w_longest = np.array([7, 3, 1] * 13)  # 15
    verifiable = list(mrz_part[:-1])  # without last character

    try:
        try:
            verifiable = ''.join([x for x in verifiable])
        except Exception:
            return None, None
        serial_int = [int(i) for i in verifiable]

        control = int(mrz_part[-1])  # last character

        verifiable_np = np.array(serial_int)
        weights = w_longest[:len(verifiable)]

        if np.sum(verifiable_np * weights) % 10 == control:
            return verifiable, True
        else:
            return verifiable, False

    except Exception:
        return verifiable, False


def mrz_fio(mrz_first: str) -> (list,list,list) or None:
    """

    :param mrz_first: first line of mrz
    :return: None or lists of characters
    """
    # -- reparation --------
    mrz_first = digits_to_letters(mrz_first)

    char_for_error = '_'

    pnrus_span = search_with_error(mrz_first, 'PNRUS')
    if pnrus_span is None:
        return None
    mrz_first = mrz_first[pnrus_span[1]:]  # del 'PNRUS'

    fio_span = search_fio_separator(mrz_first[:len(mrz_first)//2])  # search for first << or <
    # print(fio_span)
    if fio_span is None:
        return None

    mrz_f = mrz_first[:fio_span[0]]
    # -- фамилия --------
    f = []
    i = -1
    for x in mrz_f:
        i += 1  # 0
        if x in rus_codes:
            f.append(dict_codes_rus[x])
        else:
            f.append(char_for_error)
            # print('character in fion:', x, i, mrz_f)  # Debug!

    mrz_io = mrz_first[fio_span[1]:]

    # -- имя --------
    im = []
    i = -1
    for x in mrz_io:
        i += 1  # 0
        if x in rus_codes:
            im.append(dict_codes_rus[x])
        elif x == '<':
            break
        else:
            im.append(char_for_error)
            # print('character in fion:', x, i, mrz_io)  # Debug!
    # -- отчество --------
    o = []
    for x in mrz_io[i+1:]:
        i += 1
        if x in rus_codes:
            o.append(dict_codes_rus[x])
        elif x == '<':
            break
        else:
            o.append(char_for_error)
            print('character in fion:', x, i, mrz_io)  # Debug!
    return ''.join(f), ''.join(im), ''.join(o)


def mrz_roi(gray, thresh) -> list:
    """ Find ROI regions - may consist of one or two lines in one

    :param gray:
    :param thresh: exist side effect
    :return: one or two gray images of roi
    """
    height_mid = gray.shape[0] // 2
    # Remove thin lines
    linek = np.zeros((27, 27), dtype=np.uint8)
    linek[..., 13] = 1
    x = cv.morphologyEx(thresh, cv.MORPH_OPEN, linek, iterations=1)
    thresh -= x
    # img2 = cv.resize(thresh, (900, 900))
    # cv.imshow('image', img2)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q
    # img2 = cv.resize(thresh, (900, 900))
    # cv.imshow('image', img2)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q
    # gray = gray.copy()

    cnts, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                              cv.CHAIN_APPROX_SIMPLE)
    # loop over the contours
    # cnts, _ = sort_contours(contours, 'top-to-bottom')
    # gray[:][:] = 230
    # cv.drawContours(gray, contours, -1, (0, 255, 0), 3)  # DEBUG!!

    # Contours to boxes
    boxes = []
    for c in cnts:
        box = cv.boundingRect(c)
        (x, y, w, h) = box
        if y > height_mid:  # must be lower than middle
            boxes.append(box)

    box_lines = sort_contours_lines(boxes, 30, 30)  # join boxes at one line

    roi_region_arr = []
    for line in box_lines:
        for box in line:
            # print(c)
            # compute the bounding box of the contour and use the contour to
            # compute the aspect ratio and coverage ratio of the bounding box
            # width to the width of the image
            (x, y, w, h) = box#cv.boundingRect(c)
            ar = w / float(h)
            crWidth = w / float(gray.shape[1])

            # cv.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)  # DEBUG !

            # cv.imshow("ROI", img2)
            # cv.waitKey(0)

            if ar > 5 and crWidth > 0.50:
                # cv.drawContours(gray, [c], -1, (0, 255, 0), 3)  # DEBUG!!
                # print(x, y, w, h)
                # pad the bounding box since we applied erosions and now need
                # to re-grow it
                pX = int((x + w) * 0.02)  # add 1%
                pY = int((y + h) * 0.01)
                (x, y) = (x - pX, y - pY)
                (w, h) = (w + (pX * 2), h + (pY * 2))
                # extract the ROI from the image and draw a bounding box
                # surrounding the MRZ
                roi = gray[y:y + h, x:x + w].copy()
                roi_region_arr.append(roi)
                # roi, d = rotate_image(roi, get_lines_c)
                # print(d)

    # img2 = cv.resize(gray, (900,900))
    # cv.imshow("ROI", img2)
    # cv.waitKey(0)
    # print(len(roi_region_arr))
    # import random
    # r = random.randint(0, 999999999)
    # cv.imwrite("/mnt/hit4/hit4user/PycharmProjects/cnn/tmp/" + str(r) + '.png', gray)



    return roi_region_arr


def mrz(mrz_lines) -> (dict, int):
    # second line
    s_number, s_number_check = None, None
    birth_date, birth_date_check = None, None
    issue_date, code, idc_check = None, None, None
    finalok = None
    # first line
    gender = None
    mrz_f, mrz_i, mrz_o = None, None, None
    y_top = None


    try:
        l0 = None
        l1 = None
        for line in mrz_lines:
            tline = re_spaces.sub('', line[0])  # remove spaces
            if l0 is None:
                ret = search_with_error(tline, 'PNRU')  # second line must have PNRUS
                if ret and ret[0] <= 1:
                    l0 = tline
                    y_top = line[1]
                    continue
            if l1 is None:

                ret1 = re.search('^.[0-9]{3}', tline)  # first line must start with numbers
                ret2 = search_with_error(tline, 'RUS')
                if ret1 and ret2:
                    l1 = tline
                    if y_top is None:
                        y_top = line[1]

        # print("line:", mrz_lines)
        # second part
        if l1:
            mrz_s_number, mrz_s_number_check = None, None
            issue_date_and_code = None
            sn_t = letters_to_digits(l1[:10])  # 9+1
            bd_t = letters_to_digits(l1[13:20])
            idac = letters_to_digits(l1[28:43])
            gender = l1[20]
            # check sum
            if sn_t and len(sn_t) == 10:
                mrz_s_number, mrz_s_number_check = mrz_control(sn_t)  # serial number
            if bd_t and len(bd_t) == 7:
                birth_date, birth_date_check = mrz_control(bd_t)  # birth date
            if idac and len(idac) == 15:
                issue_date_and_code, idc_check = mrz_control(idac)  # last char of serial number, issue date and code
            # join
            if issue_date_and_code and len(issue_date_and_code) == 14:
                last_char_serial = issue_date_and_code[0]  # last char of serial
                issue_date = issue_date_and_code[1:7]
                code = issue_date_and_code[7:13]
                if mrz_s_number_check and mrz_s_number and len(mrz_s_number) == 9:
                    s_number = ''.join((mrz_s_number[:3], last_char_serial, mrz_s_number[3:]))
                    s_number_check = True
            # print("birth_date: ", birth_date, ok)
            # print("serial number: ", s_number, s_ok)
            # print("issue_date_and_code: ", issue_date_and_code, ok)

            try:
                # final check sum
                dop = l1[:10] + l1[13:20] + l1[21:28] + l1[28:43] + l1[43]
                _, finalok = mrz_control(dop.replace('<', '0'))
                # print('finalok: ', finalok)
            except IndexError:  # only last character was bad
                pass

        # first part
        if l0:
            ret = mrz_fio(l0)
            if ret:
                mrz_f, mrz_i, mrz_o = ret

                # print('mrz_f ', mrz_f)
                # print('mrz_i ', mrz_i)
                # print('mrz_o ', mrz_o)

    except IndexError:  # bad
        pass

    return {'s_number': s_number,
            's_number_check': s_number_check,
            'birth_date': birth_date,
            'birth_date_check': birth_date_check,
            'issue_date': issue_date,
            'code': code,
            'issue_date_and_code_check': idc_check,
            'final_check': finalok,
            'mrz_f': mrz_f,
            'mrz_i': mrz_i,
            'mrz_o': mrz_o,
            'mrz_f_check': False,
            'mrz_i_check': False,
            'mrz_o_check': False,
            'gender': gender,
            'gender_check': False}, y_top


if __name__ == '__main__':  # test
    pass
    # a = mrz_control('')
    # print(a)
