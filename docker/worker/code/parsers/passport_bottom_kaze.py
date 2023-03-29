import cv2 as cv
import pytesseract
from pytesseract import Output
from pytesseract.pytesseract import TesseractError
import re
# import matplotlib.pyplot as plt  # debug

# own
from parsers.kaze_cropper import KazeCropper
# from cnn.shared_image_functions import fix_angle, get_lines_c
from parsers import ocr
from logger import logger
# from parsers.driving_license import threashold_by_letters  # TODO: make shared
from parsers.passport_utils import search_with_error, date_re, find_date
from tesseract_lock import tesseract_image_to_data_lock



def clear_and_repeat(binary):
    """
    for /kaze_templates/passport_bottom_new.png

    :param binary: shape = (650, 795)
    """
    # clear bed text
    # print(binary.shape)
    binary[0:110, 690:] = 255  # РФ
    binary[73:116, 0:70] = 255  # фамилия
    binary[230:260, 0:25] = 255  # имя
    binary_clear = binary.copy()  # for МУЖ
    binary[301:327, 0:68] = 255  # отчество
    binary[343:398, 198:297] = 255  # Дата рождения
    binary[419:470, 0:77] = 255  # Место жительства

    binary = 255 - binary  # required for cv.BORDER_CONSTANT
    b = 5  # add border
    binary = cv.copyMakeBorder(binary, b, b, b, b, cv.BORDER_CONSTANT)
    b = 0
    for _ in range(3):  # when we repeat it start to recognize
        im1 = cv.copyMakeBorder(binary, b, b, b, b + binary.shape[1], cv.BORDER_CONSTANT)
        im2 = cv.copyMakeBorder(binary, b, b, b + binary.shape[1], b, cv.BORDER_CONSTANT)
        binary = cv.bitwise_or(im1, im2)
    binary_all = 255 - binary
    return binary_all, binary_clear


def recognize_to_lines(dict, right_limit: int, lines_diff: int) -> list:
    """
    for pytesseract.image_to_data(binary, lang='xxx', output_type=Output.DICT
    1) convert to lines
    2) convert to upper case
    """
    text_lines = []
    line = ''
    pred = 0
    # remove bad boxes
    for i in range(len(dict['level'])):
        if i > 0 and dict['top'][i-1] > dict['top'][i]:
            dict['text'][i-1] = ''

    for i in range(len(dict['level'])):
        x = dict['left'][i]
        y = dict['top'][i]
        w = dict['text'][i].upper().strip()
        if x < right_limit:
            if len(w) > 1:
                if line != '' and y - pred > lines_diff:  # end of the line
                    text_lines.append((pred, line))
                    line = ''  # clear line at the end

                pred = y

                # words to line
                if line != '':
                    line += ' ' + w  # separate with space
                else:
                    line += w

            # print(x, y, '\t\t', w)
    if line:
        text_lines.append((y, line))  # last line
    return text_lines


# def serial_number(binary):
#     # Rotate 90* left
#     binary = cv.transpose(binary)
#     binary = cv.flip(binary, flipCode=0)  # counterclockwise
#     import matplotlib.pyplot as plt
#
#     plt.imshow(binary)
#     plt.show()
#     return ''

class PassBottomCropper:
    def __init__(self):
        import os
        curr_dir = os.path.dirname(os.path.abspath(__file__))

        path_templ1 = curr_dir + '/kaze_templates/passport_bottom_new.png'  #
        path_templ2 = curr_dir + '/kaze_templates/passport_bottom_old.png'  # /mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/3/16-417-0.png

        image = cv.imread(path_templ1, cv.IMREAD_COLOR)
        cropper1 = KazeCropper(image)

        image = cv.imread(path_templ2, cv.IMREAD_COLOR)
        image = cv.fastNlMeansDenoising(image, h=8, templateWindowSize=10)  # denoise edges
        cropper2 = KazeCropper(image)
        self.croppers = (cropper1, cropper2)

    def match(self, img) -> list or None:
        """
        Apply every template and return results
        :param img:
        :return:
        """
        m_counts = list()  # list of match counts

        try:
            for c in self.croppers:
                m_counts.append(c.match(img, filt=True)[0])
        except MemoryError as e:
            return None
        return m_counts

    def crop(self, img, match_counts: list, seed:int = 1):
        """ Used after math - crop with best


            :param img:
            :param match_counts:
            :param seed: for retry
            :return: cropped image or None and cropper_id
        """

        # print(res_cr)
        max_c = max(match_counts)
        cr_index = match_counts.index(max_c)
        suitable_cropper: KazeCropper = self.croppers[cr_index]
        image = suitable_cropper.crop(img, double_crop=False, random_seed=seed)

        if image is not None:
            if cr_index == 0:
                image = image[90:740, 515:1330]  # (650, 815) #1
            elif cr_index == 1:
                image = image[55:900, 518:1315]  # ()? #2
                image = cv.resize(image, (650, 815))  # for #2
                image[700:, :] = 255  # clear bottom

        return image, cr_index


class ParserBottomKaze:
    pbc = PassBottomCropper()
    rem_punct = re.compile(r'[\.\-—]')

    # private
    @staticmethod
    def crop_and_bin(im_gray, matches, random_seed: int):
        """ For retry
        1) crop with kaze
        2) crop by template numbers
        3) binarizate
        :param im_gray:
        :param matches:
        :param random_seed:
        :return: binarized_image
        """
        img_c, ix = ParserBottomKaze.pbc.crop(im_gray, matches, seed=random_seed)  # (974, 1428) or ()?  # we will retry
        if img_c is None:
            return None
        # print(ix)
        # plt.imshow(img_c)
        # plt.show()
        # cv.imwrite("a.jpg", img_text)
        # img_text = fix_angle(img_text, get_lines_c) # may do bad thing

        bin, letters = ocr.threashold_by_letters(img_c, amount=65, scale_factor=2)  # we will retry
        # print(letters)
        # if letters and not date_re.search(letters):  # date would be recognized if we are good
        #     return None, None
        return bin

    @staticmethod
    def parse(gray):
        """
        1) AKAZE crop bottom page
        2) crop main text window
        3) fix angle of text window
        :param gray:
        :return:
        """

        ret = {'F': None,  # multiline
               'I': None,
               'O': None,
               'gender': None,
               'birth_date': None,
               'birth_place': None,  # multiline
               }

        def search_gender(xx):
            if search_with_error(xx, 'МУЖ') or search_with_error(xx, 'МУХ') or search_with_error(xx, 'МИЖ'):
                ret['gender'] = 'МУЖ'
            elif search_with_error(xx, 'ЖЕН') or search_with_error(xx, 'ХЕН'):
                ret['gender'] = 'ЖЕН'

        # obr = cv.imread(curr_dir + '/kaze_templates/passport_bottom_new.png', cv.IMREAD_GRAYSCALE)
        # obr = cv.imread('/home/u2/Pictures/bottom_passn2.png', cv.IMREAD_GRAYSCALE)
        # cropper = KazeCropper(obr)
        # 2
        # obr = cv.imread(curr_dir + '/kaze_templates/passport_bottom_new.png', cv.IMREAD_GRAYSCALE)
        # # # obr = cv.fastNlMeansDenoising(obr, h=2, templateWindowSize=10)
        # # cropper = KazeCropper(obr)
        #
        # obr = cv.imread('/home/u2/Pictures/bottom_passn1.png', cv.IMREAD_GRAYSCALE)
        # # obr = cv.fastNlMeansDenoising(obr, h=2, templateWindowSize=10)
        # cropper = KazeCropper(obr)
        # img_c = cropper.match(gray)
        # plt.imshow(img_c)
        # plt.show()
        # return ret
        #
        # cropper = KazeCropper(obr)
        matches = ParserBottomKaze.pbc.match(gray)

        binary = ParserBottomKaze.crop_and_bin(gray, matches, 1)
        if binary is None:
            # print("retry")
            binary = ParserBottomKaze.crop_and_bin(gray, matches, 2)  # retry
            if binary is None:
                return ret
        img_shape = binary.shape
        # print(img_shape)
        # binary_save = binary.copy()
        binary, b_cleared = clear_and_repeat(binary)
        # plt.imshow(b_cleared)
        # plt.show()

        try:
            # ocr_rus_strings: str = pytesseract.image_to_string(b_cleared, lang='rus',
            #                                                    config='-c tessedit_char_blacklist=' + ocr.rus_bad)
            with tesseract_image_to_data_lock:
                ocr_rus_dict: dict = pytesseract.image_to_data(binary, lang='rus', output_type=Output.DICT,
                                                               config='-c tessedit_char_blacklist=' + ocr.rus_bad)

        except TesseractError as ex:
            logger.error("ERROR: %s" % ex.message)
            return None

        # gender recognized better
        # print("gender recogn", ocr_rus_strings)
        # for x in ocr_rus_strings.split('\n'):
        #     if date_re.search(x):
        #         print("gender recogn", x)
        #         ret['birth_date'] = find_date(x)
        #         search_gender(x)

        lines = recognize_to_lines(ocr_rus_dict, img_shape[1], 20)
        F_y = 118
        I_y = 250
        O_y = 327
        gender_y = 395
        # bp_y = 468
        # bp2_y = 536
        # bp3_y = 605
        # full line height ~ 70
        # print(lines)

        if lines:
            # parse lines
            for y, line in lines:
                # print(y, line)
                if y < F_y:  # multiline
                    line = ocr.digits_to_rus_letters(line)
                    line = ParserBottomKaze.rem_punct.sub('', line).strip()
                    if ret['F'] is None:
                        ret['F'] = line
                    else:
                        ret['F'] += ' ' + line
                elif ret['I'] is None and y < I_y:
                    line = ocr.digits_to_rus_letters(line)
                    line = ParserBottomKaze.rem_punct.sub('', line).strip()
                    ret['I'] = line
                elif ret['O'] is None and y < O_y and not date_re.search(line):
                    line = ocr.digits_to_rus_letters(line)
                    line = ParserBottomKaze.rem_punct.sub('', line).strip()
                    ret['O'] = line
                elif ret['gender'] is None and y < gender_y:
                    if date_re.search(line):
                        if ret['birth_date'] is None:
                            ret['birth_date'] = find_date(line)
                        search_gender(line)
                elif gender_y-30 < y and not date_re.search(line):  # multiline
                    if ret['birth_place'] is None:
                        ret['birth_place'] = line
                    else:
                        ret['birth_place'] += ' ' + line

        # s = serial_number(binary_save)
        # print(s)

        # print(ret)

        # cv.imshow('image', binary)  # show image in window
        # cv.waitKey(0)  # wait for any key indefinitely
        # cv.destroyAllWindows()  # close window q

        # plt.imshow(binary)
        # plt.show()

        return ret


def apply_brightness_contrast(input_img, brightness=255, contrast=127):
    brightness = map(brightness, 0, 510, -255, 255)
    contrast = map(contrast, 0, 254, -127, 127)

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    # cv2.putText(buf, 'B:{},C:{}'.format(brightness, contrast), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return buf


def map(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


if __name__ == '__main__':
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples_validate/passport_and_vod/0/29-720-0.png'  # 2 words
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/3/68-469-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/3/55-456-7.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/3/12-413-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/3/11-309-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/0/3-244-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/0/11-19-0.png'

    # SEQUENCE
    p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/0/'

    import os

    pcr = PassBottomCropper()

    for i, filename in enumerate(os.listdir(p)):
        if i < 65:
            continue
        print(i, p+filename)
        image = cv.imread(p+filename, cv.IMREAD_GRAYSCALE)
        # image = apply_brightness_contrast(image, brightness=195, contrast=187)

        # m_counts: list = pcr.match(image)  # match to kaze_templates
        #
        # if m_counts is None or sum(m_counts) == 0:
        #     print("bad")
        #
        # img_cr, cr_index = pcr.crop(image, m_counts)
        # print(cr_index, m_counts)
        #
        # cv.imshow("", img_cr)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        print(ParserBottomKaze.parse(image))


    # # FILE
    # p = '/home/u2/Pictures/bottom_passn1.png'
    # p = 'kaze_templates/passport_bottom_new.png'

    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/0/6-137-0.png'
    #
    # image = cv.imread(p, cv.IMREAD_GRAYSCALE)
    # print(ParserBottomKaze.parse(image))
