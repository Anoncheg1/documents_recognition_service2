import re


class RulesOld:
    """Предположительно Приказ МВД N 782
    http://www.consultant.ru/document/cons_doc_LAW_28295/3d53b063c117e5309f5e1d78aaead678fb642834/
    """
    # old
    special_rules = {  # Е Ё  И
        'Е': {'YE': {'at_start': None, 'after': 'ЬЪАЕЁИЙОЫЭЮЯ'}},
        'Ё': {'E': {'after': 'ЧШЩЖ'},
              'YO': {'at_start': None, 'after': 'ЬЪАЕЁИЙОЫЭЮЯ'},
              'YE': {'after': 'БВГДЗКЛМНПРСТФХЦ'},
              },
        'И': {'YI': {'after': 'Ь'}}

    }
    mapping = (
        u'АБВГДЕ' + 'З' + 'ИЙКЛМНОПРСТУФ' + 'ЪЫЬЭ',  # ЕЁЖ  #И  #ХЦЧШЩ  #ЮЯ",
        u'ABVGDE' + 'Z' + 'IYKLMNOPRSTUF' + '\'Y\'e',
    )

    multi_mapping = {  # Ж ХЦЧШЩ ЮЯ
        u"Ж": u"ZH",
        u"Х": u"KH",
        u"Ц": u"TS",
        u"Ч": u"CH",
        u"Ш": u"SH",
        u"Щ": u"SHCH",
        u"Ю": u"YU",
        u"Я": u"YA",
    }


class RulesNew:
    """ Приказ МВД N 995 (2015-н/в)
     http://www.consultant.ru/document/cons_doc_LAW_195687/3197806174c701185ffd8e8986a24173958def21/

     Problem: double II may be recognized as I in pytesseract"""
    special_rules = {  # Е Ё  И
        # 'Е': {'YE': {'at_start': None, 'after': 'ЬЪ'}}, #АЕЁИЙОЫЭЮЯ
        # 'Ё': {'E': {'after': 'ЧШЩЖ'},
        #       'YO': {'at_start': None, 'after': 'ЬЪАЕЁИЙОЫЭЮЯ'},
        #       'YE': {'after': 'БВГДЗКЛМНПРСТФХЦ'},
        #       },
        # 'И': {'YI': {'after': 'Ь'}}

    }
    mapping = (
        u'АБВГДЕ' + 'З' + 'ИЙКЛМНОПРСТУФ' + 'ЫЬЭ',  # ЕЁЖ  #И  #ХЦЧШЩ  #ЮЯ",
        u'ABVGDE' + 'Z' + 'IIKLMNOPRSTUF' + 'Y\'e',
    )

    multi_mapping = {  # Ж ХЦЧШЩ ЮЯ
        u"Ж": u"ZH",
        u"Х": u"KH",
        u"Ц": u"TS",
        u"Ч": u"CH",
        u"Ш": u"SH",
        u"Щ": u"SHCH",
        u"Ю": u"IU",  #
        u"Я": u"IA",  # /mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/0/76-208-0.png
        u'Ъ': u'IE'
    }


def translit(rustr: str, rules) -> str or None:
    """
    Водительское удостоверение.

    :param rules:
    :param rustr: string
    :return: latin
    """

    special_rules = rules.special_rules
    mapping = rules.mapping
    multi_mapping = rules.multi_mapping

    new_line = rustr
    # special_rules
    for rus_char, letters_dict in special_rules.items():
        for eng_char, rules in letters_dict.items():
            for rul, support in rules.items():
                if rul == 'at_start':
                    if new_line[0] == rus_char:
                        new_line = eng_char + new_line[1:]

                if rul == 'after':
                    for _ in range(len(new_line)):
                        for i, c in enumerate(new_line):
                            if c == rus_char and i > 0 and new_line[i - 1] in support:
                                new_line = new_line[:i] + eng_char + new_line[i + 1:]
                                break

    # mapping
    for i, c in enumerate(mapping[0]):
        new_line = re.sub(c, mapping[1][i], new_line)

    # multi_mapping
    for c, repl in multi_mapping.items():
        new_line = re.sub(c, repl, new_line)

    return new_line


def check(rus, lat, p3=False) -> bool:
    """ for pytteseract comparision

    Solved problems:
     1) РЕСП, ОБЛ may be RESPULICA and OBLAST'
     2) double II may be recognized as I in pytesseract
     3) OBLAST' or OBLAST by pytesseract without '

    :param rus:
    :param lat:
    :param p3:
    :return:
    """
    # trivial
    if len(rus) <= 1 or len(lat) <= 1:
        return False

    # TODO: сделать сложный регекс чтобы исключить названия с АРЕСП и КОБЛАНКА
    # rus = re.sub(' +', '', rus)
    # lat = re.sub(' +', '', lat)
    tr_old = translit(rus, RulesOld)  # 1)
    tr_new = translit(rus, RulesNew)  # 2)
    tr_new2 = re.sub('II', 'I', tr_new)

    if p3:
        # we don't know the difference between old and new
        tr2_old = re.sub(r'RESP\.?', 'RESPUBLICA',
                         tr_old)  # new /mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/0/31-329-0.png
        tr2_old1 = re.sub(r'OBL\.?', "OBLAST'",
                          tr2_old)  # 3) new  /mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/0/37-168-0.png
        tr2_old2 = re.sub(r'OBL\.?', "OBLAST",
                          tr2_old)  # 4) without '
        tr2_new = re.sub(r'RESP\.?', 'RESPUBLIKA', tr_new)
        tr2_new1 = re.sub(r'OBL\.?', "OBLAST'", tr2_new)  # 5)
        tr2_new2 = re.sub(r'OBL\.?', "OBLAST", tr2_new)  # 6) without '

        tr2_new1_w = re.sub('II', 'I', tr2_new1)
        tr2_new2_w = re.sub('II', 'I', tr2_new2)

        alls = (tr_old, tr_new, tr_new2, tr2_old1, tr2_old2, tr2_new1, tr2_new2,
                tr2_new1_w, tr2_new2_w)

        if lat in alls:
            return True

    elif tr_old == lat or tr_new == lat or tr_new2 == lat:
        return True
    return False


def test():

    assert(check('ИРКУТСКАЯ ОБЛ', 'IRKUTSKAIA OBLAST\'', p3=True))  # without ' # /mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/26-36-0.png
    assert (check('ИРКУТСКАЯ ОБЛ', 'IRKUTSKAIA OBLAST', p3=True))  # with '
    assert (check('ЛИВИЯ', 'LIVIIA'))
    assert (check('ЛИВИЯ', 'LIVIA'))  # pytesseract mistake
    assert (check('ЛИВИЯ', 'LIVIYA'))

    assert (translit('ЕАЛЬИСЬEEЬEЯ', RulesOld) == "YEAL'YIS'EE'EYA")  # Приказ МВД N 782

    assert (translit('СЕРГЕЕВИЧ',
                     RulesOld) == 'SERGEYEVICH')  # /mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/0/45-287-0.png

    assert (translit('ИГОРЬ', RulesOld) == "IGOR'" and translit('ИГОРЬ', RulesNew) == "IGOR'")

    assert (translit('РЕСП. ДАГЕСТАН', RulesOld) == 'RESP. DAGESTAN')

    assert (translit('САГИНБАЕВ',
                     RulesNew) == 'SAGINBAEV')  # new /mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/2019080115-2-0.png

    assert (translit('ЕВГЕНЬЕВИЧ',
                     RulesOld) == "YEVGEN'YEVICH")  # 'YEVGEN'YEVICH' old # /mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/0/7-408-6.png

    assert (translit('АНАТОЛЬЕВИЧ',
                     RulesNew) == "ANATOL'EVICH")  # new /mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/30-161-10.png

    assert (check('ЧЕЛЯБИНСКАЯ ОБЛ.', 'CHELYABINSKAYA OBL.',
                  p3=True))  # /mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/0/45-176-0.png

    assert (check('ЧЕЛЯБИНСКАЯ ОБЛ', "CHELYABINSKAYA OBLAST'",
                  p3=True))  # /mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/0/45-176-0.png

    assert (check('РЕСП ДАГЕСТАН', "RESPUBLIKA DAGESTAN",
                  p3=True))  # /mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/0/45-446-0.png

    assert (check('УКРАИНА', 'UKRAINA',
                  p3=True))  # /mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/29-327-0.png

    assert (
        check('ХУССЕЙН', 'KHUSSEIN'))  # /mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/114-515-0.png


if __name__ == '__main__':  # test
    test()
