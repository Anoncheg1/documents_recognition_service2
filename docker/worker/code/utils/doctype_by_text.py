import re

import cv2 as cv
import numpy as np

# own
from parsers.ocr import image_to_string
from logger import logger as log


class DocTypes:
    consentToProcessingOfPersonalData = 10  # Согласие на обработку персональных данных
    individualConditions = 46  # Индивидуальные условия
    loginAssignmentNotice = 45  # Извещение о присвоении логина
    applicationForOnlineServices = 44  # Заявление о подключении онлайн-сервисов РУСНАРБАНК
    applicationForAutocredit = 43  # Заявление о предоставлении кредита
    applicationForAdvanceAcceptance = 42  # Заявление о заранее данном акцепте ФЛ
    applicationForTransferringFromAccountFL = 41  # Заявление на перевод денежных средств со счета ФЛ
    borrowerProfile = 40  # Анкета клиента - заемщика банка ( Анкета заемщика )
    telemedicine = 60  #
    zajOPrisoedKDogovKBO = 61 # ЗАЯВЛЕНИЕ О ПРИСОЕДИНЕНИИ К ДОГОВОРУ КОМПЛЕКСНОГО БАНКОВСКОГО ОБСЛУЖИВАНИЯ ФИЗИЧЕСКИХ ЛИЦ


# format: ((must be), (several mustbe), (must not be):(id, description)
dict_docs = {
    # Файлы андерайтера
    (("ЗАКЛЮЧЕНИЕПОКРЕДИТНОЙЗАЯВКЕ",), (), ()): (80, 'Заключение андеррайтера'),
    (("ПРОФЕССИОНАЛЬНОЕСУЖДЕНИЕПООЦЕНКЕКРЕДИТНОГОРИСКАПОССУДЕ",), (), ()): (
        1, 'Профессиональное суждение по оценке кредитного риска по ссуде'),
    (("ОЦЕНКАФИНАНСОВОГОПОЛОЖЕНИЯ",), (), ()): (2, 'Оценка финансового положения'),

    # Сделка
    ((), ("АСИЕЗАЕМЩИКАН", "НАОБРАБОТКУМОИХПЕРСОНАЛЬНЫХДАННЫХ"),
     ("УСЛОВИЕОБУСТУПКЕ", "ЦЕЛИИСПОЛЬЗОВАНИЯЗАЕМЩИКОМ")): (
        10, 'Согласие на обработку персональных данных'),
    (("ДОГОВОР", "КУПЛИПРОДАЖИ"), (),
     ("ПРИЛОЖЕНИЕ", "ОСНОВАНИЕПОДОГОВОРУ", "ОПЛАТАПОДОГОВОРУ", "СОГЛАСНОДОГОВОРУ", "СЧЕТНАОПЛАТУ",
      "ЗАЕМЩИКСОГЛАСЕНСТЕМЧТОВСЛУЧ", "АЕНЕСОГЛАСИЯЗАЧЕРКНУТЬ",  # последняя страница 501
      "ПОРЯДОКПРИЕМАПЕРЕДАЧИТРАНСПОРТНОГОСРЕДСТВА", "СТВАНЕПРЕОДОЛИМОЙСИЛ", "ПРИНЦИПАЛОБЯЗАН",
      "ДОПОЛНИТЕЛЬНЫЕУСЛОВИЯ")): (
        11, 'Договор купли продажи'),
    (("ДОГОВОРКОМИССИИ", "ПРЕДМЕТДОГОВОРА"), (),
     ("ПРИЛОЖЕНИЕ", "ОСНОВАНИЕПОДОГОВОРУ", "ОПЛАТАПОДОГОВОРУ", "СОГЛАСНОДОГОВОРУ", "СЧЕТНАОПЛАТУ")): (
        12, 'Договор комиссии'),
    (("АГЕНТСКИЙДОГОВОР", "ПРЕДМЕТДОГОВОРА"), (),
     ("ПРИЛОЖЕНИЕ", "ОСНОВАНИЕПОДОГОВОРУ", "ОПЛАТАПОДОГОВОРУ", "СОГЛАСНОДОГОВОРУ", "СЧЕТНАОПЛАТУ")): (
        13, 'Агентский договор'),
    (("АКТПРИЕМАПЕРЕДАЧИ",), (), ("ПОРЯДОКПРИЕМАПЕРЕДАЧИ", "ФОРСМАЖОР", "ПОРЯДОКИЗМЕНЕНИЯИДОПОЛНЕНИЯДОГОВОРА",
                                  "ПОРЯДОКРАЗРЕШЕНИЯСПОРОВ")): (
        20, 'Акр приема-передачи'),
    (("ПРИЕМОСДАТОЧНЫЙАКТ",), (), ()): (21, 'Приемо-сдаточный акт'),
    (("СЧЕТ№",), ("ПОЛУЧАТ", "УЧАТЕЛЬ", "ПОСТАВЩИК", "ОПЛАТА"), ()): (22, 'Счет'),
    (("СЧЕГ№",), ("ПОЛУЧАТ", "УЧАТЕЛЬ", "ПОСТАВЩИК", "ОПЛАТА"), ()): (22, 'Счет'),  # recognition mistake
    (("СЧЕТНАОПЛАТУ№", "ИНН"), (), ()): (22, 'Счет'),
    (("КВИТАНЦИЯ", "РИНЯТООТ"), (), ()): (
        23, 'Квитанция'),
    (("ПОЛИС№", "КАСКО", "ДОГОВОРАСТРАХОВАНИЯ", "СТРАХОВАТЕЛЬ"), (), ()): (
        24, 'Страховой полис КАСКО'),
    (("ПОЛИСКОМПЛЕКСНОГОСТРАХОВАНИЯТРАНСПОРТНОГОСРЕДСТВА", "СТРАХОВАТЕЛЬ"), (), ()): (
        24, 'Страховой полис КАСКО (КТС)'),
    (("ДОГОВОРПОРУЧЕНИЯ",), (),
     ("ПРИЛОЖЕНИЕ", "ОСНОВАНИЕПОДОГОВОРУ", "ОПЛАТАПОДОГОВОРУ", "СОГЛАСНОДОГОВОРУ", "СЧЕТНАОПЛАТУ")): (
        26, 'Договор поручения'),
    (("ВЫПИСКАИЗЭЛЕКТРОННОГОПАСПОРТА",), (), ()): (27, 'Выписка из электронного паспорта'),
    (("АНКЕТАКЛИЕНТАФИЗИЧЕСКОГОЛИЦА",), (), ()): (28, 'Анкета клиента физлица у партнера'),
    (("ЭЛЕКТРОННАЯКАРТА", "НАДОРОГЕ", "ПОДДЕРЖК"), (), ()): (29, 'Карта технической помощи на дороге'),

    # Сгенерированные документы
    (("АНКЕТАЗАЕМЩИКАФИЗИЧЕСКОГОЛИЦА",), (), ()): (40, 'Анкета клиента - заемщика банка'),
    (("ЗАЯВЛЕНИЕНАПЕРЕВОДДЕ",), (), ()): (41, 'Заявление на перевод денежных средств со счета ФЛ'),
    (("ЗАЯВЛЕНИЕОЗАРАНЕЕДАН",), (), ()): (42, 'Заявление о заранее данном акцепте ФЛ'),
    (("ЗАЯВЛЕНИЕОПРЕДОСТАВЛЕНИИКРЕДИТА",), (), ()): (43, 'Заявление о предоставлении кредита'),
    (("КПРАВИЛАМДИСТАНЦИОННОГОБАНКОВСКОГООБСЛУЖИВАНИЯ",), (), ()): (
        44, 'Заявление о подключении онлайн-сервисов РУСНАРБАНК'),
    (("ЛОГИНАДЛЯДОСТУПАКСИСТЕМЕ",), (), ()): (45, 'Извещение о присвоении логина'),
    (("ПОЛНАЯСТОИМОСТЬКРЕДИТА", "ИНДИВИДУАЛЬНЫЕУСЛОВИЯ"), (), ()): (46, 'Индивидуальные условия'),
    (("РАСПОРЯЖЕНИЕОПРЕДОСТАВЛЕНИИКРЕДИТА",), (), ()): (47, 'Распоряжение о предоставлении кредита'),
    (("СЕРТИФИКАТНАСТОЯЩИМСЕРТИФИКАТОМКОММЕРЧЕСКИЙБ",), (), ()): (48, 'Одобрение - сертификат'),
    # ((,), (), ()): (49, 'Согласие на обработку персональных данных - руснарбанк'),
    (("СПОСОБЫПОГАШЕНИЯКРЕДИТА",), (), ()): (50, 'Способы погашения кредита'),
    (("ГАРАНТИЙНОЕПИСЬМО",), (), ()): (
        51, 'Гарантийное письмо - в автосалон за автомобиль/за дополнительное оборудование'),
    (("СПИСАНОСОСЧПЛА", "ПОРУЧЕНИЕ"), (), ()): (
        53, 'Платежное поручение - Основной платежный документ/на оплату КАСКО/на оплату дополнительного оборудования'),
    (("ПОЛИССТРАХОВАНИЯЖИЗНИ",), (), ()): (56, 'Полис страхования от несчастного случая'),
    (("ПОЛИССТРАХОВАНИЯОТПОТЕРИРАБОТЫ",), (), ()): (57, 'Полис страхования от потери работы'),
    (("ПАМЯТКАКДОГОВОРАМЛИЧНОГОСТРАХОВАНИЯ",), (), ()): (58, 'Памятка к договорам личного страхования'),
    (("ЗАЯВЛЕНИЕНАУЧАСТИЕВГОСУДАРСТВЕННОЙПРОГРАММЕ",), (), ()):
        (59, 'Заявление на участие в государственной программе'),  # applicationForStateSupport
    ((), (), ()):
        (60, 'ТЕЛЕМЕДИЦИНА'),  # telemedicine
    (("ЗАЯВЛЕНИЕОПРИСОЕДИНЕНИИКДОГОВОРУКОМПЛЕКСНОГОБАНКО",), (), ()):
        (61, 'ЗАЯВЛЕНИЕ О ПРИСОЕДИНЕНИИ К ДОГОВОРУ КОМПЛЕКСНОГО БАНКОВСКОГО ОБСЛУЖИВАНИЯ ФИЗИЧЕСКИХ ЛИЦ'),
    #
    ((), ("ПАСПОРТТРАН", "ГОСРЕДСТ", "ОСОБЫЕОТМЕТКИ", "ИДЕНТИФИКАЦИОННЫЙНО", "ГОСУДАРСТВЕННЫЙРЕГИСТРАЦИОННЫЙЗНАК",
          "ВЫДАНОГИБДД"), ()): (110, 'ПТС front. Пасспорт транспортного средства - первая.'),
    ((), ("ОСОБЫЕОТМЕТКИ", "СВИДЕТЕЛЬСТВООРЕГИСТРАЦИИТ", "РЕГИСТРАЦИОННЫЙЗНАК", "ДАТАРЕГИСТРАЦИИ",
          "ДАТАПРОДАЖИПЕРЕДАЧИ"), ()): (110, 'ПТС back. Пасспорт транспортного средства - вторая.'),

    # документы партнеров
    (("ТИПЗАПРАШИВАЕМОГОКРЕДИТА", "ЗАПРАШИВАЕМАЯСУММАКРЕДИТА"), ("АВТОСПЕЦЦЕНТР", "РОЛЬВПРЕДПОЛАГАЕМОЙСДЕЛКЕ"), ()):
        (501, 'Анкета-заявление на предоставление кредита. АвтоСпецЦентр'),

    # капитал-лайф
    (("КАПИТАЛ", "ЗАЯВЛЕНИЕОСТРАХОВАНИИ", "МЕННЫЙЗАПРОС"), (), ()): (510, 'КАПЛАЙФ Заявление о страховании'),
    # капитал-лайф письменный запрос Страховщика)
    (("КАПИТАЛ", "ПОЛИС№"), (), ()): (511, 'КАПЛАЙФ Заявление о страховании'),
    (('ТАБЛИЦАРАЗМЕРОВСТРАХОВЫХСУММ',), (), ()): (512, 'Таблица размеров страховых сумм'),
    (('КАПИТАЛ', 'ПАМЯТКАВАЖНЫЕПОЛОЖЕНИЯ'), (), ()): (513, 'КАПЛАЙФ Памятка Важные положения договора страхования'),
    (('КАПИТАЛ',), ('ПРОГРАММАДОБРОВ', 'ОЛЬНОГОИНДИВИДУ', 'АЛЬНОГОСТРАХ', 'ОВАНИЯЗАЁМЩ', 'ИКОВКРЕДИТАДЛЯКЛИЕНТОВ'), ()):
        (514,
         'КАПЛАЙФ Программа добровольного индивидуального страхования заёмщиков кредита для клиентов АО КБ «РУСНАРБАНК»'),
    # СОГАЗ
    (('СОГАЗ'), ('НАСТОЯЩИМПО', 'ЛИСОМОФЕРТОЙДА', 'ПОЛИСОФЕРТАСТРАХОВАНИЯОТН', 'ЕСЧАСТНЫХСЛУЧАЕВИБОЛЕЗНЕЙ'), ()): (
    520, 'СОГАЗ Заявление о страховании (полис-оферта)'),
    (('ТАБЛИЦАСТРАХОВЫХВЫПЛАТПРИТЕЛЕСНЫХПОВРЕЖДЕНИЯХ', 'ПРИЛОЖЕНИЕ'),
     ('ЗАСТРАХОВАННОГОЛИЦАВРЕЗУ', 'ЛЬТАТЕНЕСЧАСТНОГОСЛУЧАЯ'), ()):
        (521, 'СОГАЗ Таблица страховых выплат при телесных повреждениях'),
    (('СОГАЗ', 'ПАМЯТКА'), ('ПОЛИСОФЕРТАСТРАХОВАНИЯОТН', 'ЕСЧАСТНЫХСЛУЧАЕВИБОЛЕЗНЕЙ'), ()):
        (522, 'СОГАЗ Памятка к заявлению о страховании'),

}

docs_types = sorted(dict_docs.items(), key=lambda x: len(x[0][0]), reverse=True)  # asc longest to shortest

re_ftext = re.compile(r'[^А-Яа-я№]*')

# type:
not_multi = 0
multifile = 1
fixed_files = 2
# doc_id:(type, files_count)
multifile_doctypes = {80: (2, 2),
                      1: (0,),
                      2: (0,),
                      10: (0,),
                      11: (1,),
                      12: (1,),
                      13: (1,),
                      20: (0,),
                      21: (0,),
                      22: (0,),
                      23: (0,),
                      24: (1,),
                      26: (1,),
                      27: (0,),
                      28: (1,),
                      29: (1,),
                      40: (2, 2),
                      41: (0,),
                      42: (0,),
                      43: (0,),
                      44: (0,),
                      45: (0,),
                      46: (1,),
                      47: (0,),
                      48: (0,),
                      # 49: (0,),
                      50: (0,),
                      51: (0,),
                      53: (0,),
                      56: (2, 3),
                      58: (0,),

                      110: (0,),
                      501: (0,),
                      510: (0,),
                      511: (1,),
                      512: (0,),
                      513: (0,),
                      514: (1,),
                      520: (1,),
                      521: (0,),
                      522: (0,),
                      }

# doc types that has multiple files
multifile_types = {k: v for k, v in multifile_doctypes.items() if v[0] == 1 or v[0] == 2}

debug_type = []  # debug


def _get_type_by_text(text) -> tuple or None:
    """
    Метод для получения типа страницы документа

    :returnint: Отдает номер распознанного типа документа или None
    """
    global dictionary_getter, debug_type

    for (mb_strings, several_strings, mnb_strings), d_type in docs_types:
        must_be = [text.find(s) != -1 for s in mb_strings]  # 1)
        if not all(must_be):
            continue
        must_not_be = [text.find(s) != -1 for s in mnb_strings]  # 2)
        if any(must_not_be):
            continue
        several = [text.find(s) != -1 for s in several_strings]  # 3)
        # Test!
        # if d_type[0] == 110:
        #     debug_type.append(several)
        if all(must_be) and not any(must_not_be) and several.count(True) >= round(len(several) / 2):  # 2 of 3
            return d_type
    return None


def _fast_cropper(image):
    """ 0.0001 < x < 0.9
    и Квитанция 0.3 < x < 0.7 если нашлась похожая contourArea"""
    total_area = image.shape[0] * image.shape[1]
    gray = image.copy()
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ret2, r = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(r, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    boxes = []
    for i in range(len(contours)):
        x, y, w, h = cv.boundingRect(contours[i])
        c = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]])  # rectangle back to contour
        a = cv.contourArea(c)
        if (total_area * 0.0001) < a < (total_area * 0.9):
            boxes.append([x, y, x + w, y + h])
    boxes = np.asarray(boxes)
    if boxes.size != 0:
        x = np.min(boxes[:, 0])  # left
        y = np.min(boxes[:, 1])  # top
        x_w = np.max(boxes[:, 2])  # right
        y_h = np.max(boxes[:, 3])  # bottom
        # cv.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        # img = image
        if (total_area * 0.7) > (x_w * y_h) > (total_area * 0.3):  # квитанция size
            y -= int(image.shape[1] * 0.02)
            if y < 0:
                y = 0
            x -= int(image.shape[0] * 0.02)
            if x < 0:
                x = 0
            y_h += int(image.shape[1] * 0.02)
            x_w += int(image.shape[0] * 0.02)
            img = r[y:y_h, x:x_w].copy()
            return img
    # crop header
    h_height = int(image.shape[0] // 2.7)
    img = r[0:h_height].copy()
    return img


def simple_detect_type(image: np.array, text_doc_orientation=None) -> (int, str or None, int or None):
    """

    :param image:
    :param text_doc_orientation:
    :return:
    """
    global re_ftext

    log.debug("simple_detect_type started")

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # if we got orientation from pages before this
    if text_doc_orientation is not None:
        for i in range(text_doc_orientation):
            gray = np.rot90(gray)
        # crop
        img = _fast_cropper(gray)
        # get text
        t = image_to_string(img, 'rus')
        text = re_ftext.sub('', t).upper()  # remove ,.-
        doc_id = _get_type_by_text(text)
        if doc_id is not None:
            return doc_id[0], doc_id[1], text_doc_orientation
    else:
        text = ''
        # Rotate
        for i in range(4):  # 0,1,2,3
            # image rotate not clockwise
            if i != 0:
                gray = np.rot90(gray)

            # crop
            img = _fast_cropper(gray)
            # get text
            t = image_to_string(img, 'rus')
            # TESTING!!
            # print('as', t)
            if t is not None:
                t = re_ftext.sub('', t).upper()  # remove ,.-
                # TESTING!!
                # print('as2', t)
                # tmp = cv.resize(img, (900, 900))
                # cv.imshow('image', tmp)  # show image in window
                # cv.waitKey(0)  # wait for any key indefinitely
                # cv.destroyAllWindows()  # close window

                text += t
                # text to doc id
                doc_id = _get_type_by_text(text)
                if doc_id is not None:
                    return doc_id[0], doc_id[1], i

        return 0, None, None


def mark_second_files(final_list: list):
    """
    Mark second pages of documents in result pages
    Used in final_json_processing before samplify_api - progress, page_completed, final_ready

    :param final_list: [{},{'document_type': 40, }, ...]
    :return: final_list with replaced 'document_type' 110 to 1000 for second files
    """
    mul: dict = multifile_types

    new_final = []
    pred = None
    count_second = 0
    for x in final_list:
        if x is not None:  # do not clear pred
            if 'document_type' in x:
                dtype = x['document_type']  # current doc type

                # dtype with low probability and pred was
                if (dtype in (130, 140, 100) and (x['qc'] == 3 or x['qc'] == 4)) \
                        and pred is not None \
                        and ((mul[pred][0] == 1) or (mul[pred][0] == 2 and count_second < mul[pred][1])):  # type 1 or 2
                    count_second += 1
                    new_page = {'document_type': pred, "description": "second page", "qc": 0}
                    if 'file_uuid' in x:  # save file_uuid
                        new_page['file_uuid'] = x['file_uuid']
                    new_final.append(new_page)
                    continue
                elif dtype in mul:
                    pred = dtype
                    count_second = 1
                else:
                    pred = None
        # if not continue
        new_final.append(x)
    return new_final


if __name__ == '__main__':  # test
    pass
