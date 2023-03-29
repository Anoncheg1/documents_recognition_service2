# MRZ - machine-readable zone
# ROI - region of interest OpenVC
# https://www.pyimagesearch.com/2015/11/30/detecting-machine-readable-zones-in-passport-images/
import cv2 as cv
import re
import imutils
import inject
# own
from parsers.passport_mrz import mrz_roi, mrz
from parsers.passport_utils import roi_ocr
from parsers.passport_main_page import main_two_pages
from parsers.passport_contours import contours
from parsers.passport_serial import serial_n
from doc_types import DocTypeDetected
from utils.progress_and_output import OutputToRedis
from logger import logger as log

# test
# from groonga import FIOChecker
# fio_checker = FIOChecker(10)


fio_checker = inject.instance('FIOChecker')


def passport_preprocess(gray):
    """
    :param gray:
    :return: (None, None, None, None)
    """
    mrz_lines = None  # mrz strings
    cnts_save = None  # contours for main page
    thresh_save = None  # binary for main page
    roi_regions = None
    for i in range(5):

        cnts, thresh = contours(gray.copy(), i)  # gradient by X

        if cnts_save is None and cnts is not None and 50 < len(cnts) < 95:
            cnts_save = cnts
            thresh_save = thresh

        if not mrz_lines:
            roi_regions: list = mrz_roi(gray, thresh.copy())  # get regions MRZ
            if roi_regions:  # not empty
                # MRZ get text in regions ENG - better numbers  OCRB - better names
                ret = roi_ocr(roi_regions, lang='ocrb')  # , white_list=passport_mrz.chars)
                if len(ret[0]) >= 1:
                    ret = [(l, y) for l, y in zip(ret[0], ret[1]) if len(l) > 40]  # skip short lines (44 exactly)
                    if len(ret) >= 1:
                        mrz_lines = ret
        if cnts_save and mrz_lines:
            break
    return mrz_lines, cnts_save, thresh_save, roi_regions


def passport_main_page(obj: DocTypeDetected):
    """ Do not read other things if MRZ was fully readed

    :param img: not resized, cropped, fixed rotation and orientation, BGR
    :return: object with OUTPUT_OBJ property
    """
    log.debug("passport_main_page parser started")

    img = obj.not_resized_cropped

    class anonymous_return:  # always
        OUTPUT_OBJ: dict = {'qc': 4,
                            'MRZ': None,  # may be None
                            'main_top_page': None,  # may be None
                            'main_bottom_page': None,  # may be None
                            'serial_number': None,  # may be None
                            'serial_number_check': False,
                            'suggest': None  # may be None
                            }

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    height = int(1995.8816568047337)  # average prep
    width = int(1555.0118343195265)  # average
    gray = imutils.resize(gray, width=width)  # resized

    mrz_lines, cnts_save, thresh_save, roi_regions = passport_preprocess(gray)
    OutputToRedis.progress(obj, 30)

    # debug
    # print(cnts_save)
    # cv.drawContours(img, cnts_save, -1, (0, 255, 0), 3)
    # img2 = cv.resize(img, (900, 900))
    # cv.imshow('image', img2)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q

    # -- Machine Readable Zone --
    mrs_s_number = None
    mrz_f, mrz_i, mrz_o = None, None, None
    y_top = None
    mrz_gender = None
    if mrz_lines:
        mrz_re, y_top = mrz(mrz_lines)  # MRZ return STRUCTURE inside!
        anonymous_return.OUTPUT_OBJ['MRZ'] = mrz_re
        mrz_f = mrz_re['mrz_f']
        mrz_i = mrz_re['mrz_i']
        mrz_o = mrz_re['mrz_o']
        mrs_s_number = mrz_re['s_number']
        mrz_gender = mrz_re['gender']

    OutputToRedis.progress(obj, 35)

    # -- Main page --
    (f_pass, i_pass, o_pass) = None, None, None
    gender_pass = None
    if cnts_save:
        # 'main_top_page' and 'main_bottom_page' STRUCTURE inside!
        mtp, mbp = main_two_pages(cnts_save, gray, thresh_save, y_top, obj)
        anonymous_return.OUTPUT_OBJ['main_top_page'] = mtp
        anonymous_return.OUTPUT_OBJ['main_bottom_page'] = mbp
        (f_pass, i_pass, o_pass) = mbp['F'], mbp['I'], mbp['O']
        gender_pass = mbp['gender']

    OutputToRedis.progress(obj, 90)

    # --  MRZ-FIO compare with bottom page --
    (f_check, i_check, o_check) = False, False, False  # return
    if mrz_lines:
        if mrz_f and mrz_f == f_pass and len(mrz_f) > 1:
            f_check = True
        if mrz_i and mrz_i == i_pass and len(mrz_i) > 1:
            i_check = True
        if mrz_o and mrz_o == o_pass and len(mrz_o) > 1:
            o_check = True
        anonymous_return.OUTPUT_OBJ['MRZ']['mrz_f_check'] = f_check
        anonymous_return.OUTPUT_OBJ['MRZ']['mrz_i_check'] = i_check
        anonymous_return.OUTPUT_OBJ['MRZ']['mrz_o_check'] = o_check

    # --  MRZ-gender compare with bottom page --
    if mrz_gender and (mrz_gender == 'F' and gender_pass == 'ЖЕН') or (mrz_gender == 'M' and gender_pass == 'МУЖ'):
        anonymous_return.OUTPUT_OBJ['MRZ']['gender_check'] = True

    # -- Passport serial number and code
    # mrz serial number is short we need red serial number
    ret = serial_n(gray)  # get serial_number
    if ret:
        s_number, s_checked = ret

        # CHECK SERIAL NUMBER by comparing with half-readed mrz
        if s_number:
            if not s_checked and mrs_s_number:
                # lets delete last number of serial code to compare with mrz
                if s_number == mrs_s_number:
                    s_checked = True
            # print('serial_number=', ret_s_number[0:2], ret_s_number[2:4], ret_s_number[4:], 'checked=', s_checked)
            anonymous_return.OUTPUT_OBJ['serial_number'] = s_number
            anonymous_return.OUTPUT_OBJ['serial_number_check'] = s_checked

    OutputToRedis.progress(obj, 99)

    # -- code_pod_check (if issue_date_and_code_check=False)
    mtp = anonymous_return.OUTPUT_OBJ['main_top_page']
    if mtp is not None:
        mtp['code_pod_check'] = False
        if mtp['code_pod'] is not None and anonymous_return.OUTPUT_OBJ['MRZ'] is not None:
            filtered_code_pod = re.sub('[^0-9]', '', mtp['code_pod'])
            if filtered_code_pod == anonymous_return.OUTPUT_OBJ['MRZ']['code']:
                mtp['code_pod_check'] = True

    # --  Groonga FIO suggestion

    fio_ch = {
        'F': None,
        'F_gender': None,
        'F_score': 0,
        'I': None,
        'I_gender': None,
        'I_score': 0,
        'O': None,
        'O_gender': None,
        'O_score': 0
    }
    # f_pass, i_pass, o_pass:
    # 'mrz_f': mrz_f,            'mrz_i': mrz_i,            'mrz_o': mrz_o,
    res_i, res_o, res_f = None, None, None
    # name
    if i_pass and anonymous_return.OUTPUT_OBJ['MRZ'] and anonymous_return.OUTPUT_OBJ['MRZ']['mrz_i']:
        res_i = fio_checker.double_query_name(anonymous_return.OUTPUT_OBJ['MRZ']['mrz_i'], i_pass)
    elif i_pass:
        res_i = fio_checker.wrapper_with_crop_retry(fio_checker.query_name, i_pass)
    elif anonymous_return.OUTPUT_OBJ['MRZ'] and anonymous_return.OUTPUT_OBJ['MRZ']['mrz_i']:
        res_i = fio_checker.wrapper_with_crop_retry(fio_checker.query_name,
                                                    anonymous_return.OUTPUT_OBJ['MRZ']['mrz_i'])
    # patrony TODO: double quert.
    if anonymous_return.OUTPUT_OBJ['MRZ'] and anonymous_return.OUTPUT_OBJ['MRZ']['mrz_o']:
        res_o = fio_checker.wrapper_with_crop_retry(fio_checker.query_patronymic,
                                                    anonymous_return.OUTPUT_OBJ['MRZ']['mrz_o'])
    elif o_pass:
        res_o = fio_checker.wrapper_with_crop_retry(fio_checker.query_patronymic, o_pass)
    # surname
    if f_pass and anonymous_return.OUTPUT_OBJ['MRZ'] and anonymous_return.OUTPUT_OBJ['MRZ']['mrz_f']:
        res_f = fio_checker.double_query_surname(anonymous_return.OUTPUT_OBJ['MRZ']['mrz_f'], f_pass)
    elif f_pass:
        res_f = fio_checker.wrapper_with_crop_retry(fio_checker.query_surname, f_pass)
    elif anonymous_return.OUTPUT_OBJ['MRZ'] and anonymous_return.OUTPUT_OBJ['MRZ']['mrz_f']:
        res_f = fio_checker.wrapper_with_crop_retry(fio_checker.query_surname,
                                                    anonymous_return.OUTPUT_OBJ['MRZ']['mrz_f'])

    if res_i:
        fio_ch['I'], fio_ch['I_gender'], fio_ch['I_score'] = res_i
    if res_o:
        fio_ch['O'], fio_ch['O_gender'], fio_ch['O_score'] = res_o
    if res_f:
        fio_ch['F'], fio_ch['F_gender'], fio_ch['F_score'] = res_f

    if res_i or res_o or res_f:
        anonymous_return.OUTPUT_OBJ['suggest'] = fio_ch

    # SET QUALITY OF RECOGNITION
    r = anonymous_return.OUTPUT_OBJ
    if r['MRZ'] is None and r['main_top_page'] is None \
            and r['main_bottom_page'] is None and r['serial_number'] is None:
        quality = 3
    else:
        if r['MRZ'] is None and r['main_top_page'] is not None and r['main_bottom_page'] is not None:
            values = list(anonymous_return.OUTPUT_OBJ['main_top_page'].values()) + list(anonymous_return.OUTPUT_OBJ[
                                                                                            'main_bottom_page'].values())
            if values.count(None) > len(values) / 2:
                quality = 2
            else:
                quality = 1

        elif r['MRZ'] is not None and r['MRZ']['final_check']:
            quality = 0
        else:
            quality = 2

    anonymous_return.OUTPUT_OBJ['qc'] = quality
    return anonymous_return


if __name__ == '__main__':

    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/prep/passport/passport_main/'
    # w, h = list(), list()
    # for i, filename in enumerate(os.listdir(p)):
    #     im = cv.imread(p + filename)
    #     w.append(im.shape[0])
    #     h.append(im.shape[1])
    # print(np.average(w))
    # print(np.average(h))
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_mestgit/1/83-215-3.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_mestgit/1/'+'50-292-1.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_mestgit/1/'+'50-292-1.png'
    # p = '/tmp/32-163-0.png'
    # p = '/tmp/3-11-0.png'
    # p = '/tmp/Паспорт_1.JPG'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/prep/passport/passport_main/58--0.png_0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/prep/passport/passport_main/65--0.png_0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/prep/passport/passport_main/81-213-0.png_0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/prep/passport/passport_main/48-59-0.png_0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/prep/passport/passport_main/58-356-0.png_0.png' # bad
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/prep/passport/passport_main/30-328-0.png_0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/prep/passport/passport_main/31-273-0.png_0.png'
    # p = '/home/u2/Desktop/1/Паспорт_1.JPG'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/prep_res_pas/passport/passport_main/49-347-0.png_3.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/prep/passport/passport_main/56--0.png_0.png'

    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/prep/passport/passport_main/3-11-0.png_0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/0/3-11-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/prep_nores/passport/passport_main/32-42-0.png_0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/prep_nores/passport/passport_main/74-372-0.png_2.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/prep_nores/passport/passport_main/79-377-0.png_3.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/unknown/0/1-1-2.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/0/6-137-0.png'
    # p ='/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/0/7-248-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/0/8-249-0.png'

    # FOLDER
    # # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples_validate/passport_and_vod/0/'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples_validate/passport/passport_main/0/'
    # import os
    #
    # from cnn.shared_image_functions import crop
    #
    # for i, filename in enumerate(os.listdir(p)):
    #     if i < 3:
    #         continue
    #     print(i, p + filename)
    #     img = cv.imread(p + filename, cv.IMREAD_COLOR)
    #
    #     # lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    #     # lab_planes = cv.split(lab)
    #     # clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(100, 100))
    #     # lab_planes[0] = clahe.apply(lab_planes[0])
    #     # lab = cv.merge(lab_planes)
    #     # img = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    #     # img = apply_brightness_contrast(img, brightness=255, contrast=167)
    #
    #     # from skimage.filters import threshold_yen
    #     # from skimage.exposure import rescale_intensity
    #     # from skimage.io import imread, imsave
    #     # img = imread(p + filename)
    #     # yen_threshold = threshold_yen(img)
    #     # bright = rescale_intensity(img, (0, yen_threshold), (0, 255))
    #     # imsave('out.tiff', bright)
    #     # img = cv.imread('out.tiff', cv.IMREAD_COLOR)
    #
    #     img, _ = crop(img, rotate=True, rate=1)
    #     s = passport_main_page(img)
    #     print(s.OUTPUT_OBJ)
    # tmp = cv.resize(img, (300, 300))
    # cv.imshow('image', tmp)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q
    #     s = passport_main_page(img)
    #     print(s.OUTPUT_OBJ)

    # FILE
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples_validate/passport_and_vod/0/130-821-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples_validate/passport_and_vod/0/264-956-1.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples_validate/passport_and_vod/0/3-694-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples_validate/passport_and_vod/0/29-720-0.png'  # 2 words
    p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/0/12-253-0.png'
    img = cv.imread(p, cv.IMREAD_COLOR)
    from cnn.shared_image_functions import crop

    img, _ = crop(img, rotate=True, rate=1)
    for _ in range(4 - 3):
        timg = cv.transpose(img)
        img = cv.flip(timg, flipCode=1)

    tmp = cv.resize(img, (900, 900))
    cv.imshow('image', tmp)  # show image in window
    cv.waitKey(0)  # wait for any key indefinitely
    cv.destroyAllWindows()  # close window q
    #
    s = passport_main_page(img)
    print(s.OUTPUT_OBJ)

    # for _ in range(4 - 1):
    #     timg = cv.transpose(img)
    #     img = cv.flip(timg, flipCode=1)
    # cv.imwrite('/tmp/Паспорт_1.JPG', img)
    # img2 = cv.resize(img, (900, 900))

    # # print(im)
    # cv.imwrite('/tmp/3-11-0.png', img)
    #
    # img = cv.imread(p)

    # img = cv.resize(img, (width, height))
    # for i in range(9):
    #     res = page_sep(img, 9-i)
    #     if res is not None:
    #         break
    # print(res)

    # s = passport_main_page(img)
    # print(s.OUTPUT_OBJ)
    # print('\n')
    # print(s.OUTPUT_OBJ)

    # activered = a[0] - a[2]
    # gray = cv.fastNlMeansDenoising(activered, h=120, templateWindowSize=40)  # h=40, templateWindowSize=20)
    # gray = cv.erode(gray, None, iterations=4)
    # gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    # cnts = cv.findContours(gray.copy(), cv.RETR_EXTERNAL,
    #                        cv.CHAIN_APPROX_SIMPLE)[-2]
    # cv.drawContours(img, cnts, -1, (0, 255, 0), 3)

    # src = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # dst = cv.equalizeHist(src)
    # gamma = 1.0
    # lookUpTable = np.empty((1,256), np.uint8)
    # for i in range(256):
    #     lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    # res = cv.LUT(img, lookUpTable)
    # a = cv.split(img)
    # print(a[3].shape)
    # a[2] = cv.convertScaleAbs(a[2], beta=30.5)
    # res = cv.merge(a)

    # main_page(img)
    # print(img.shape)

    # img2 = cv.resize(img, (900, 900))
    # import matplotlib.pyplot as plt
    # plt.imshow(img2)
    # plt.show()
    # cv.imshow('image', img2)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q
    # im = cropimg(img, document=False)
    # im, _ = rotate_image(im, get_lines_c2)

    # img2 = cv.resize(im, (900, 900))
    # cv.imshow('image', img2)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # ret2, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # mb  = MRZBoxLocator()
    # box :RotatedBox = mb.__call__(th)[0]
    #
    # th = box.extract_from_image(img=img)
    # # roi = gray[y:y + h, x:x + w].copy()
    # # p = read_mrz(p, save_roi=True)
    # print(box)
    # img2 = cv.resize(th, (900, 900))
    # cv.imshow('image', img2)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q

    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/prep/passport/passport_main/'
    # for i, filename in enumerate(os.listdir(p)):
    #     if i < 95:
    #         continue
    #     print("i:", i, p + filename)
    #     img = cv.imread(p + filename)
    #     img = cv.resize(img, (width, height))
    #     res = None
    #     y1,y2 = None, None
    #     for i in range(9):
    #         res = page_sep(img, 9-i)
    #         if res is not None:
    #             break
    #     if res is not None:
    #         if res[3] < 100:
    #             y1 = res[1]
    #             y2 = res[1] + 100
    #         else:
    #             # print(res[0])
    #             y1 = res[1]
    #             y2 = res[1] + res[3]
    #     else:
    #         pass
    #         # y1 = h/2
    #         # y2 = h/2
    #     print(y1, y2)

    #     # if i < 32:  # 28 34 bad 35 no first
    #     if i < 163:  # 22 worst
    #         continue
    #     im = cv.imread(p + filename, cv.IMREAD_COLOR)
    #
    #     im, _ = crop_passport(im, rotate=True)
    #     # img = im
    #
    #     # print("wtf")
    #     # im, _ = rotate_image(im, get_lines_c2)
    #     print("/tmp/" + filename)
    #     cv.imwrite("/tmp/" + filename, im)
    #     # main_page(im)
    #     # im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    #     # # im = cv.fastNlMeansDenoising(im, h=20, templateWindowSize=10)
    #     # im = cv.threshold(im, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    #     img2 = cv.resize(im, (900, 900))
    #
    #     cv.imshow('image', img2)  # show image in window
    #     cv.waitKey(0)  # wait for any key indefinitely
    #     cv.destroyAllWindows()  # close window q
    #
    # tessdata_dir_config = '--tessdata-dir "/mnt/hit4/hit4user/PycharmProjects/rec2/"'
    #
    # letters = pytesseract.image_to_boxes(img, lang='rus') #, lang='ocrb', config=tessdata_dir_config)
    # # letters = pytesseract.image_to_string(im, lang='rus')  # , lang='ocrb', config=tessdata_dir_config)
    # # print(letters)
    # letters = letters.split('\n')
    # letters = [letter.split() for letter in letters]
    # h, w = im.shape
    # for letter in letters:
    #     cv.rectangle(img, (int(letter[1]), h - int(letter[2])), (int(letter[3]), h - int(letter[4])), (0, 0, 255), 2)
    # img2 = cv.resize(im, (900, 900))
    #
    # cv.imshow('image', img2)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q

    # mrz = read_mrz(p + filename, save_roi=True)
    # p = MRZPipeline(p + filename, extra_cmdline_params='--oem 0')
    # print(p.result)
    # img2 = cv.resize(im, (900, 900))
    #
    # cv.imshow('image', img2)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q
    # p = '/home/u/PycharmProjects/rec2/cnn/samples/passport/passport_main/0/105-237-0.png'
    # p = '/dev/shm/tmp.png'
