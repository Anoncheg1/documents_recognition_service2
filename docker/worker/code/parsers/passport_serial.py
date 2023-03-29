import cv2 as cv
#own
from parsers.passport_utils import sort_contours_lines, enlarge_boxes, box_to_text
from parsers.passport_contours import contours


def serial_n(gray):
    """
    Used in passport
    """
    # Rotate 90* left
    gray = cv.transpose(gray)
    gray = cv.flip(gray, flipCode=0)  # counterclockwise

    cnts_save = None
    for i in range(5):
        cnts, thresh = contours(gray.copy(), i)  # to gray and get contours
        if cnts and 50 < len(cnts) < 95:
            cnts_save = cnts
            break

    if cnts_save:
        ret_s_number, s_checked = serial_number(cnts_save, gray)
        return ret_s_number, s_checked
    else:
        return None


def serial_number(cnts, gray):
    """
    TODO: return left and right?

    :param cnts:
    :param gray:
    :return:
    """

    h_img, w_img = gray.shape

    hs = h_img // 110
    ws = w_img // 110  # min limits
    hh = h_img // 19  # hight limit

    w_mid = w_img // 2

    boxes_filtered = []
    for contour in cnts:
        x, y, w, h = cv.boundingRect(contour)
        # Don't plot small false positives that aren't text
        if w < ws or h < hs or h > hh:  # or w > wh:  # or h < 10 or w > 900:
            continue
        # cv.drawContours(gray, [contour], -1, (0, 255, 0), 3)
        boxes_filtered.append((x, y, w, h))

    boxes_filtered = enlarge_boxes(boxes_filtered, ws, hs)
    lines = sort_contours_lines(boxes_filtered, ws, hs)

    serial_num = None
    ok: bool = False
    for i_line, b_line in enumerate(lines[:2]):
        line_p1 = ''
        line_p1_l = 0
        line_p2 = ''
        line_p2_l = 0

        # join rectangles on one page - for better recognition
        b1 = [box for box in b_line if box[0] < w_mid]
        if b1:
            b1_rec = (min([box[0] for box in b1]),
                      min([box[1] for box in b1]),
                      (b1[-1][0] + b1[-1][2]) - b1[0][0],
                      (b1[-1][1] + b1[-1][3]) - b1[0][1])
            # left page
            line_p1, line_p1_l = box_to_text(b1_rec, gray)

        b2 = [box for box in b_line if box[0] > w_mid]
        if b2:
            b2_rec = (min([box[0] for box in b2]),
                      min([box[1] for box in b2]),
                      (b2[-1][0] + b2[-1][2]) - b2[0][0],
                      (b2[-1][1] + b2[-1][3]) - b2[0][1])

            # right page
            line_p2, line_p2_l = box_to_text(b2_rec, gray)

        # counting result
        # print("elen", line_p1_l, line_p2_l)
        if len(line_p1) == 10 and len(line_p2) == 10 and line_p1 == line_p2:
            ok = True
            serial_num = line_p1
        elif len(line_p1) == 10 and line_p1_l >= line_p2_l:
            serial_num = line_p1
        elif len(line_p2) == 10 and line_p2_l >= line_p1_l:
            serial_num = line_p2
        if serial_num:
            break

    # print(line_p1, "|", line_p2)
    #
    # img2 = cv.resize(gray, (900, 900))
    # cv.imshow('image', img2)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q

    return serial_num, ok