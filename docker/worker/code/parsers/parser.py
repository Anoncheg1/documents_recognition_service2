from parsers.passport import passport_main_page
from parsers.driving_license import dl_parse
from utils.doctype_by_text import docs_types, DocTypes
from doc_types import DocTypeDetected
from concurrent.futures import ThreadPoolExecutor
from utils.progress_and_output import PROFILING
from parsers.handwritings.parser import parser_handwritings
# from multiprocessing.pool import Pool


def unknown(obj: DocTypeDetected):
    class anonymous_return:
        OUTPUT_OBJ: dict = {}

    anonymous_return.OUTPUT_OBJ['qc'] = 4
    return anonymous_return


def skip(obj: DocTypeDetected):
    class anonymous_return:
        OUTPUT_OBJ: dict = {}

    anonymous_return.OUTPUT_OBJ['qc'] = 0
    return anonymous_return


def common(obj: DocTypeDetected):
    class anonymous_return:
        OUTPUT_OBJ: dict = {}

    anonymous_return.OUTPUT_OBJ['qc'] = 0
    return anonymous_return


def passport_and_drivelicense(obj: DocTypeDetected):
    """
    side effect: obj.doc_type, obj.description

    :param obj: img_raw, img_cropped
    :return:
    """
    class anonymous_return:
        OUTPUT_OBJ: dict = {'qc': 4,
                            'passport_main': None,
                            'driving_license': None
                            }

    # parsers
    if PROFILING:
        ret_passp = passport_main_page(obj)
        ret_drlic = dl_parse(obj)
    else:
        # Multi-Processes
        # executor = Pool(processes=2)
        # passport_main_page_process = executor.apply_async(passport_main_page, args=(obj,))
        # dl_parse_thread = executor.apply_async(dl_parse, args=(obj,))
        # ret_passp = passport_main_page_process.get()
        # ret_drlic = dl_parse_thread.get()
        # Multi-Threading
        with ThreadPoolExecutor(max_workers=2) as executor:  # memory leaking
            passport_main_page_thread = executor.submit(passport_main_page, obj)
            dl_parse_thread = executor.submit(dl_parse, obj)
            ret_passp = passport_main_page_thread.result()
            ret_drlic = dl_parse_thread.result()

    # result of parsers
    if ret_passp.OUTPUT_OBJ['qc'] == 4 and ret_drlic.OUTPUT_OBJ['qc'] == 4:
        anonymous_return.OUTPUT_OBJ = {'qc': 4}
        anonymous_return.OUTPUT_OBJ.update(ret_passp.OUTPUT_OBJ)
        obj.doc_type = 100
        obj.description = 'Passport: main page'
    elif ret_drlic.OUTPUT_OBJ['qc'] == 4:
        anonymous_return.OUTPUT_OBJ = ret_passp.OUTPUT_OBJ
        obj.doc_type = 100
        obj.description = 'Passport: main page'
    elif ret_passp.OUTPUT_OBJ['qc'] == 4:
        anonymous_return.OUTPUT_OBJ = ret_drlic.OUTPUT_OBJ
        obj.doc_type = 130
        obj.description = 'Driving license'
    else:
        anonymous_return.OUTPUT_OBJ['passport_main'] = ret_passp.OUTPUT_OBJ
        anonymous_return.OUTPUT_OBJ['driving_license'] = ret_drlic.OUTPUT_OBJ
        anonymous_return.OUTPUT_OBJ['qc'] = (ret_passp.OUTPUT_OBJ['qc'] + ret_drlic.OUTPUT_OBJ['qc']) // 2

    return anonymous_return


def handwriting_parser(obj: DocTypeDetected):
    res = parser_handwritings(obj.original, obj.doc_type, obj.page)
    if res is None:
        qc = 1
    else:
        qc = 0
    class Anunumous:
        OUTPUT_OBJ = {'qc': qc, 'text_doc_fields': res}
    return Anunumous


method_number_list = {
    0: unknown,
    # 0: predict_doctype,
    # 1: OKUD.OKUD_0710001,
    # 2: OKUD.OKUD_0710001_CONTINUEPAGE,
    # 3: OKUD.OKUD_0710002,
    # 4: OKUD.OKUD_0710002_CONTINUEPAGE,
    # 5: OKUD.OKUD_ALTERCLASS, # (документов, которые нельзя идентифицировать по маркеру) - продолжение предыдущего
    14: common,  # simple_detect_type was successful
    100: passport_main_page,  # cropped one image
    101: skip,  # 101-9 - passport other
    110: skip,  # pts,
    120: skip,  # photo
    130: dl_parse,  # Driving license. accept raw one image
    140: passport_and_drivelicense  # Driving license and Passport not main page. Accept (raw image, cropped)
}

method_number_list.update({x[1][0]: common for x in docs_types})
method_number_list.update({x: handwriting_parser for x in (
    DocTypes.consentToProcessingOfPersonalData,  # 10 - 2
    DocTypes.individualConditions,  # 46 - 3
    DocTypes.loginAssignmentNotice,  # 45 - 4
    # DocTypes.applicationForOnlineServices,  # 44 - 5
    DocTypes.applicationForAutocredit,  # 43 - 6
    DocTypes.applicationForAdvanceAcceptance,  # 42 - 7
    DocTypes.applicationForTransferringFromAccountFL,  # 41 - 8
    DocTypes.borrowerProfile,  # 40 - 9
)})


# if __name__ == '__main__':
    # import cv2 as cv
    #
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/4-302-0.png'
    # img = cv.imread(p, cv.IMREAD_COLOR)
    # print(img.shape)
    # img_passport, _ = crop(img, rotate=True, rate=1)
    # print(img.shape)
    # # print(passport_and_drivelicense(img).OUTPUT_OBJ)
