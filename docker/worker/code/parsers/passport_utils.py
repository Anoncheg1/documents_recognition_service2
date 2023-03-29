import cv2 as cv
import re
# own
from parsers import ocr
from parsers.ocr import roi_preperation

# alphabet = 'АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЭЮЯЬЪвгджзийклмнопрстуфхцчшщыэюяьъ'  # без 'абе' - АБЕ в регистрах не похожи

memo = {}  # levenshtein


def levenshtein(s: str, t: str) -> int:
    """
    Side effect to memo!

    :param s:
    :param t:
    :return: 0 - len(s)
    """
    if s == "":
        return len(t)
    if t == "":
        return len(s)
    cost = 0 if s[-1] == t[-1] else 1

    i1 = (s[:-1], t)
    if not i1 in memo:
        memo[i1] = levenshtein(*i1)
    i2 = (s, t[:-1])
    if not i2 in memo:
        memo[i2] = levenshtein(*i2)
    i3 = (s[:-1], t[:-1])
    if not i3 in memo:
        memo[i3] = levenshtein(*i3)
    res = min([memo[i1] + 1, memo[i2] + 1, memo[i3] + cost])

    return res


def enlarge_boxes(boxes, ws, hs):
    for i, b in enumerate(boxes):
        x, y, w, h = b
        x = x - round(ws * 2.2)
        y = y - round(hs * 1.5)
        h = h + round(hs * 2.2)
        w = w + ws * 4

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        boxes[i] = (x, y, w, h)
    return boxes


def roi_ocr(region_array: list, lang='ocrb', debug=False, white_list: str = None, blacklist: str = None) -> (list, list):
    """ Optical Character Recognition in region of interest

    :param debug:
    :param lang:
    :param white_list:
    :param region_array: gray
    :return: typle of (text list and y list) of or empty lists
    """
    global re_spaces
    ret_text = []
    ret_y = []
    for r in region_array:
        for i in range(10):  # recognition loop
            r = roi_preperation(r, i, scale_factor=1)
            if debug:
                from matplotlib import pyplot as plt
                plt.imshow(r)
                plt.show()

                # cv.imshow('', r)
                # cv.waitKey(0)  # wait for any key indefinitely
                # cv.destroyAllWindows()  # close window

            text = ocr.image_to_string(r, lang=lang, white_list=white_list, blacklist=blacklist)

            # text = re_spaces.sub('', text)
            if text:
                text_a = text.split('\n')
                for x in text_a:
                    if x != '':
                        y = cv.boundingRect(r)[1]
                        ret_text.append(x)
                        ret_y.append(y)

                if ret_text:
                    break
    return ret_text, ret_y


def sort_contours_lines(boundingBoxes, ws, hs) -> list:
    """ by lines top-bottom, left right

    :param cnts:
    :return: (cnts, boundingBoxes)
    """

    # sort boxes by lines
    lines = list()
    # del_centers = list()
    for b in boundingBoxes:
        x, y, w, h = b
        # print(x, y, w, h)
        center = y + h / 2
        # print(c_center)
        foundline_flag = False
        for l in lines:  # search in already existed lines
            # print("wtf")
            if y < l[0] < y + h:
                new_c = (l[0] + center) / 2
                l[0] = new_c
                l[1].append(b)

                foundline_flag = True

                break
        if not foundline_flag:  # new line
            lines.append([center, [b]])

    lines = sorted(lines, key=lambda x: x[0], reverse=False)  # top-to-bottom

    s_lines = list()
    for line in lines:

        sorted_l = sorted(line[1], reverse=False)  # left-to-right

        # Join Close Rectangles
        del_indexes = []
        for i, x in enumerate(sorted_l):
            if i > 0:
                pred = sorted_l[i - 1]

                overlap = pred[0] + pred[2] - x[0]  # + overlap, - gap  # if o< 0 = gap if o > 0 = overlap

                # if right and left side is close or overlap
                if (-overlap < (ws * 2)
                    or overlap > 0) \
                        and abs(pred[3] - x[3]) < hs:  # not big difference in height
                    # new x = pred
                    x_new = list(pred)
                    # new y = highest
                    if pred[1] < x[1]:
                        x_new[1] = pred[1]
                    else:
                        x_new[1] = x[1]
                    # new widh = x + width
                    # 1) gap = w1 + w2 + gap
                    # 2) overlap = w1 + w2 - overlap
                    # 3) one in another - leave larges

                    if overlap > 0 and (x[0] + x[2]) < (pred[0] + pred[2]):
                        x_new[2] = pred[2]
                    else:
                        x_new[2] = round(pred[2] + x[2] - overlap)
                    # new height
                    x_new[3] = round((pred[3] + x[3]) / 1.9)

                    sorted_l[i] = tuple(x_new)  # replace x
                    del_indexes.append(i - 1)  # remove pred after loop
        for i in reversed(del_indexes):
            del sorted_l[i]
        s_lines.append(sorted_l)

        #
        # if (x < l_center and x+h > l_center):
        #     print("ok")
        # if l_center == 0 or not (x < l_center and x+h > l_center):  # если строка пустая
        #     print("wtf")
        #     l_center = c_center  # new line
        #     lines.append([c])
        # else:
        #     print("wtf2")
        #     lines[-1].append(c)  # add to last string
        #     l_center = (l_center + c_center) // 2

    # for k, v in lines.items():
    #     print(v)
    #     cnts, _ = sort_contours(v, "left-to-right")
    #     lines[k] = cnts

    return s_lines
    # return sorted_l


# razd = r'(( [\.,—\-:])|([\.,—\-:] )|([\.,—\-: ]))'
razd = r'(( [\.—\-])|([\.—\-] )|([\.—\- ]))'  # with whitelist

razd_sub_re = re.compile(razd)
date_re = re.compile(r'[0-3][0-9]' + razd + r'?[0-1][0-9]' + razd + r'?(19|20)[0-9]{2}')
date2_re = re.compile(r'([^' + ocr.rus_alph + ']|^)((19|20)[0-9]{2})([^0-9]|$)')


def find_date(line: str):
    """Extract: 13.13.2017 or 2017

    :param line:
    :return: str date or None
    """
    date = None
    d = date_re.search(line)  # full date
    year = date2_re.search(line)  # only year
    # year = re.search(r'([^' + alphabet + ']|^)((19|20)[0-9]{2})([^0-9]|$)', line)
    if d or year:
        if d:
            sp = d.span()
        else:
            sp = year.span(2)
        date = line[sp[0]:sp[1]]
        # print(date)
        date = razd_sub_re.sub('.', date)  # sepparators must be .
        if len(date) > 4:  # add .
            if date[2] != '.':
                date = date[:2] + '.' + date[2:]
            if date[5] != '.':
                date = date[:5] + '.' + date[5:]

    return date


# passport serial number
def most_common_substring10(text: str) -> str:
    """ if text is None - exception"""
    a = []
    for j in range(10):
        for i in range(7):
            if (i * 10 + 10 + j) <= len(text):
                a.append(text[i * 10 + j:i * 10 + 10 + j])
    v = {x: a.count(x) for x in a if a.count(x) > 2}  # all 10chars windows with more then 2 occurance
    l = list(v.keys())
    if l and l[0] == text[0:10]:  # first sequence have largest priority - we take it if exist
        return l[0]
    re = {k: levenshtein(k * 8, text) for k in v.keys()}  # sequence: compare_result
    if re:
        return sorted(re, key=re.__getitem__)[0]  # ascending - clothest to text
    else:
        return ''


def search_with_error(source: str, sub: str) -> tuple or None:
    """
    with one char error - char added or replaced

    :param source: where to find
    :param sub: what to find
    :return: (start,end) or None
    """
    source = source.upper()
    sub = sub.upper()

    l = list(sub)
    for i in range(1, len(sub)):  # middle error P.?NRUS PN.?RUS PNR.?US PNRU.?S PNRUS.?
        reg = l[:i] + list('.?') + l[i:]
        reg = ''.join(reg)
        # print(reg)
        f = re.search(reg, source)
        if f:
            return f.span()

    for i in range(len(sub)):  # char error .?NRUS P.?RUS PN.?US PNR.?S PNRU.?
        reg = l[:i] + list('.?') + l[(i+1):]
        reg = ''.join(reg)
        # print(reg)
        f = re.search(reg, source)
        if f:
            return f.span()

    return None


def box_to_text(box, gray) -> (str, int):
    """TODO: clear comments

    :param box:
    :param gray:
    :return:
    """
    x, y, w, h = box
    # cv.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Debug!
    roi = gray[y:y + h, x:x + w]
    # tesseract we repeat image for
    # text, _ = roi_ocr([roi], lang='eng', debug=False, white_list=ocr.numbers)  # get text in regions
    # cv.imshow('image', roi)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q
    # cv.imwrite('/home/u2/Documents/a.jpg', roi)

    _, im = cv.threshold(roi, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # to binary
    im = 255 - im  # required for cv.BORDER_CONSTANT
    b = 5  # add border
    im = cv.copyMakeBorder(im, b, b, b, b, cv.BORDER_CONSTANT)
    b = 0  #
    for _ in range(3):  # when we repeat symbol it start to recognize it
        # im1 = cv.copyMakeBorder(im, b, b, b, b, cv.BORDER_CONSTANT)  # add height to top
        # im2 = cv.copyMakeBorder(im, b, b + im.shape[0], b, b, cv.BORDER_CONSTANT)  # add height to bottom
        im1 = cv.copyMakeBorder(im, b, b, b, b + im.shape[1], cv.BORDER_CONSTANT)
        im2 = cv.copyMakeBorder(im, b, b, b + im.shape[1], b, cv.BORDER_CONSTANT)
        im = cv.bitwise_or(im1, im2)  # merge
    im = 255 - im
    # cv.imshow('image', im)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q
    line = ocr.image_to_string(im, lang='eng', white_list=ocr.numbers)
    if line:
        line = re.sub(r"\D", "", line)
        text = most_common_substring10(line)
        return text, len(line)
    return '', 0

    # line = pytesseract.image_to_string(im, lang='ocrb', config=c_string)
    # return text
    # if len(text) > 0:
    #     # t = letters_to_digits(' '.join(text))
    #     # # LEGACY - replaced with tesseract whitelist
    #     # t = t.replace('B', '8')
    #     # t = t.replace('C', '0')
    #     # t = t.replace('D', '0')
    #     # t = t.replace('U', '0')
    #     # t = re.sub(r'[^0-9]', '', t)
    #     # t = re.sub(r' +', '', t)
    #     return text
    # else:
    #     return ''


def test():
    assert (find_date('13.32.2017') == '2017')
    assert (find_date('1312.2017') == '13.12.2017')
    assert (find_date('1312 .2017') == '13.12.2017')
    assert (find_date('4.6.2077') == '2077')
    assert (razd_sub_re.sub('', '02 0 031328') == '020031328')
    assert (levenshtein('ВОДИТЕЛЬСКОЕ', 'ВОДИТЕЛЬСКОЕ') == 0)
    assert (find_date('42320774') is None)
    assert (most_common_substring10('') == '')
    assert (most_common_substring10('3105591400310559140031055914003105591400310559140031055914003105591400310559140')
            == '3105591400')

    assert (most_common_substring10('73172248147317224814131722481413172248141317224814131722481473172248141317224814')
            == '7317224814')
    assert (search_with_error('.PcRUS', 'PNRUS') == (1, 6))


if __name__ == '__main__':  # test
    test()
