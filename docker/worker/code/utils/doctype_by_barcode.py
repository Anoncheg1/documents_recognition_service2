import cv2 as cv
import pyzbar.pyzbar as pyzbar
import numpy as np

from logger import logger as log
from utils.doctype_by_text import dict_docs
from cnn.shared_image_functions import rotate

doc_types: dict = {x: y for x, y in dict_docs.values()}


def read_barcode(image: np.array) -> (int, str or None, int or None):
    """
    version 1
    19 - version 1
    9 - sepparator

    :param image:
    :return: doctype: int, description: str, page: int
    """

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def dec(gray):
        if gray.shape[0] < 1000:
            # 0.022 - profiling
            gray_resized = cv.resize(gray, None, fx=2.9, fy=2.9, interpolation=cv.INTER_CUBIC)
            return pyzbar.decode(gray_resized, symbols=[pyzbar.ZBarSymbol.I25])
        else:
            return pyzbar.decode(gray, symbols=[pyzbar.ZBarSymbol.I25])

    barcodes = dec(gray)

    angle = 4

    # Fast rotate to find out barcode
    hard_readable1, hard_readable2, hard_readable3 = False, False, False
    doc_type, description, page, hard_readable1 = parse_barcodes(barcodes)
    if doc_type == 0:
        rot_gray = rotate(gray, angle, 0.95)
        barcodes = dec(rot_gray)
        doc_type, description, page, hard_readable2 = parse_barcodes(barcodes)
        if doc_type == 0:
            rot_gray = rotate(gray, -angle, 0.95)
            barcodes = dec(rot_gray)
            doc_type, description, page, hard_readable3 = parse_barcodes(barcodes)
            if doc_type == 0 and any((hard_readable1, hard_readable2, hard_readable3)) is False:
                return 0, None, None

    # Slow rotation for better recognition
    if doc_type == 0:  # and any((hard_readable1, hard_readable2, hard_readable3)) is True:
        angles = (-3.8, -3.5, -3.2, -2.9, -2.6, -2.3, -2. , -1.7, -1.4, -1.1, -0.8,
       -0.5, -0.2,  0.1,  0.4,  0.7,  1. ,  1.3,  1.6,  1.9,  2.2,  2.5,
        2.8,  3.1,  3.4,  3.7)  # np.arange(-3.9, 4, 0.2)  np.arange(-3.8, 4, 0.3)
        for angle in angles:
            rot_gray = rotate(gray, -angle, 0.95)
            barcodes = dec(rot_gray)
            doc_type, description, page, _ = parse_barcodes(barcodes)
            if doc_type != 0:
                break

    return doc_type, description, page

    # print(barcodes[0])

    # tmp = cv.resize(gray, (900, 900))
    # cv.imshow('image', tmp)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window


def parse_barcodes(barcodes) -> tuple:
    """

    :param barcodes:
    :return:
    doc_type, - 0 or >0
    description, None or string
    page, None or >0
    hard_readable - bool flag - barcode was detected with error
    """
    hard_readable = False
    if barcodes:
        for bar in barcodes:
            data = bar.data.decode('ascii')
            if not data:
                log.warn("read_barcode: fail to decode data in ASCII " + str(barcodes))
                continue

            # decode base-8 numeric system
            # print("1", data)

            if bar.type != 'I25':
                log.debug("barcode type is not I25")
                continue

            # old format 019912919146 with 0
            if data[0] == '0':
                data = data[1:]
            data_ch_i25 = data[-1]
            data = data[:-1]  # remove I25 checksum
            checksum_i25 = 0
            for i, x in enumerate(data):  # first must be 0
                if i % 2 == 1:
                    checksum_i25 += int(x)
                else:
                    checksum_i25 += int(x) * 3

            checksum_i25 = 10 - (checksum_i25 % 10)
            if checksum_i25 == 10:
                checksum_i25 = 0
            if checksum_i25 != int(data_ch_i25):
                log.debug("read_barcode: bad native checksum of I25 barcode: " + str(barcodes))
                hard_readable = True
                continue
            # print("2", data)
            split = data.split('9')
            if len(split) != 5:
                log.error("read_barcode: I25 barcode can not be split according to version! " + str(barcodes))
                hard_readable = True
                continue
            try:
                version, reserv, doctype, page, checksum = split  # TODO: accept anather version
                version = int(version, base=8)
                if reserv:
                    reserv = int(reserv, base=8)
                else:
                    reserv = 0
                doctype = int(doctype, base=8)
                page = int(page, base=8)
                checksum = int(checksum, base=8)
                # print('dec:',version, reserv, doctype, page)
                sum = version + reserv + doctype + page
                # print(sum)
            except ValueError:
                log.error("read_barcode: ValueError! " + str(barcodes))
                hard_readable = True
                continue

            if sum % 64 != checksum:
                log.error("read_barcode: bad second checksum of I25 barcode! " + str(barcodes))
                hard_readable = True
                continue

            if doctype not in doc_types:
                log.critical("read_barcode: doctype {} not in doc types {}".format(str(doctype), str(barcodes)))
                continue
            return doctype, doc_types[doctype], page, None
    return 0, None, None, hard_readable


if __name__ == '__main__':  # test
    # files
    from os import listdir, path
    # import UtilModule
    from utils.pdf_convertors import pdf2png
    import logging
    from logger import logger as log
    log.setLevel(logging.DEBUG)


    # folder with pdf
    # mypath = '/home/u2/Downloads/templ3/'
    # files = [path.join(mypath, f) for f in listdir(mypath)]
    files = ['/home/u2/Downloads/фин.pdf']
    print(files)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirpath:
        for i, x in enumerate(files):
            print(i, x)
            # if i < 1 or i > 2:
            #     continue
            # fs = UtilModule.UtilClass.PdfToPng(x, '../1')
            if x.endswith('pdf'):
                fs = pdf2png(x, tmpdir=tmpdirpath)  # split
            else:
                fs = [x]
            print(fs)
            for j, fp in enumerate(fs):
                if j < 2 or j > 2:
                    continue
                print(fp)
                img = cv.imread(fp, cv.IMREAD_COLOR)
                # img = np.rot90(img)
                # img = np.rot90(img)
                res = read_barcode(img)
                print(i, j, res, fp)
