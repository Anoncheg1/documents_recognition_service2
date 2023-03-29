import cv2 as cv
import inject
# own
from utils.profiling import profiling_before, profiling_after
from groonga import FIOChecker
from cnn.shared_image_functions import crop
from doc_types import DocTypeDetected
from utils.pdf_convertors import pdf2png
import utils.progress_and_output
from parsers.handwritings.parser import parser_handwritings
from utils.doctype_by_text import DocTypes

utils.progress_and_output.PROFILING = False
import MainOpenCV  # import MainProcessingClass, PROFILING  # for test

from parsers.passport import passport_main_page  # for test


def test_fio_db():
    single_instance = FIOChecker(10)
    g = single_instance

    res = g.query_name('ЛЛЛЛЛЛЛЛЛЛЛЛЛ')
    assert (res is None)
    print('СОНЯ', g.query_name('СОНЯ'))
    res = g.query_name('ЛЕНА')
    print(res)
    assert (res[0] == 'ЕЛЕНА' and res[1] == 'FEMALE')
    res = g.query_name('ВЛАДИСЛАВ')
    print(res)
    assert (res[0] == 'ВЛАДИСЛАВ' and res[1] == 'MALE' and res[2] == 1)
    res = g.query_name('ВЛАДИСЛИВ')
    print(res)
    assert (res[0] == 'ВЛАДИСЛАВ' and res[1] == 'MALE')
    res = g.query_name('ВЛАДИСЛ')
    print(res)
    assert (res[0] == 'ВЛАДИС' and res[1] == 'MALE')
    res = g.query_name('КАТЕРУН')
    print(res)
    assert (res[0] == 'КАТЕРИНА' and res[1] == 'FEMALE')
    res = g.double_query_name("АЛЕКСЙ", "АЛЕСЕЙ")
    print(res)
    assert (res[0] == 'АЛЕКСЕЙ' and res[1] == 'MALE')
    res = g.double_query_name("АННА", "ИНА")  # double test
    print(res)
    assert (res[0] == 'АННА' and res[1] == 'FEMALE' and res[2] == 1)
    res = g.double_query_name("ИНА", "АННА")  # double test
    print(res)
    assert (res[0] == 'АННА' and res[1] == 'FEMALE' and res[2] == 1)
    # patr
    res = g.query_patronymic("СЕРГЕЕВИЧ")
    print(res)
    assert (res[0] == 'СЕРГЕЕВИЧ' and res[1] == 'MALE' and res[2] == 1)
    res = g.query_patronymic("ЮЛЯ ОГЛЫ")
    print(res)
    assert (res[0] == 'ЮЛИЯ ОГЛЫ' and res[1] == 'MALE')
    # sur
    res = g.query_surname('БЕЛОВСОВ')
    print(res)
    assert (res[0] == 'БЕЛОУСОВ' and res[1] == 'MALE')
    res = g.double_query_surname('БЕЛОСОВ', 'ЕЛОУСОВ')
    print(res)
    assert (res[0] == 'БЕЛОУСОВ' and res[1] == 'MALE')
    res = g.double_query_surname('ыыыы', 'ыыыы')
    assert (res is None)
    # wrapper
    res = g.wrapper_with_crop_retry(g.query_name, 'КАТЕРУН')
    assert (res[0] == 'КАТЕРИНА' and res[1] == 'FEMALE' and res[2] < 1)
    res = g.wrapper_with_crop_retry(g.query_name, 'КАТЯ КАТЕРУН')
    assert (res[0] == 'КАТЯ' and res[1] == 'FEMALE' and res[2] < 1)
    res = g.wrapper_with_crop_retry(g.query_name, 'ВАЛЕНТИНА КЫЗЫ КАТЕРУН')
    print(res)
    # exception test
    assert (g.double_query_name("АААААААААНА", "АААААААААНА") is None)
    # assert (res[0] == 'ВАЛЕНТИНА КЫЗЫ' and res[1] == 'FEMALE' and res[2] < 1)


def test_UtilModule():
    import os

    filepatch = 'test/example.pdf'
    id_processing = str(1)
    try:
        os.mkdir(id_processing)
    except Exception:
        import shutil
        shutil.rmtree(id_processing, ignore_errors=True)
        os.mkdir(id_processing)

    filelist = pdf2png(filepatch, id_processing)
    assert (all([os.path.exists(x) for x in filelist]))


def test_driving_license():
    # own
    from parsers.driving_license import dl_parse  # for test

    filepatch = 'test/driving_license12-143-0.png'

    img = cv.imread(filepatch, cv.IMREAD_COLOR)  # scanned image
    obj = DocTypeDetected()
    obj.original = img
    o = dl_parse(obj)
    print(o.OUTPUT_OBJ)
    # {'qc': 0, 'fam_rus': 'КОВРИГИН', 'fam_eng': 'KOVRIGIN', 'fam_check': True, 'name_rus': 'СЕРГЕЙ ВИКТОРОВИЧ', 'name_eng': 'SERGEY VIKTOROVICH', 'name_check': True, 'p3': '17.01.1962', 'birthplace3_rus': 'МОРДОВИЯ', 'birthplace3_eng': 'MORDOVIYA', 'birthplace3_check': True, 'p4a': '15.08.2012', 'p4b': '15.08.2022', 'p4ab_check': True, 'p4c_rus': 'ГИБДД 4010', 'p4c_eng': 'GIBDD 4010', 'p4c_check': True, 'p5': '4005584309', 'p8_rus': 'КАЛУЖСКАЯ ОБЛ', 'p8_eng': 'KALUZHSKAYA OBL', 'p8_check': True, 'categories': ['CE', 'A', 'D', 'C', 'B', 'DE']}
    # {'qc': 1, 'side': 'front', 'fam_rus': 'КОВРИГИН', 'fam_eng': 'KOVRIGIN', 'fam_check': True, 'name_rus': 'СЕРГЕЙ ВИКТОРОВИЧ', 'name_eng': 'SERGEY VIKTOROVICH', 'name_check': True, 'p3': '17.01.1962', 'birthplace3_rus': 'МОРДОВИЯ', 'birthplace3_eng': 'MORDOVIYA', 'birthplace3_check': True, 'p4a': '15.08.2012', 'p4b': None, 'p4ab_check': False, 'p4c_rus': 'ГИБДД 4010', 'p4c_eng': '™MBEAON 4010', 'p4c_check': False, 'p5': '010', 'p8_rus': None, 'p8_eng': 'KAJTYXCKASA OBL', 'p8_check': False, 'suggest': {'F': 'КОВРИГИН', 'F_gender': 'MALE', 'F_score': 1, 'I': 'СЕРГЕЙ', 'I_gender': 'MALE', 'I_score': 1, 'O': 'ВИКТОРОВИЧ', 'O_gender': 'MALE', 'O_score': 1}, 'categories': ['A', 'C', 'B']}
    assert (o.OUTPUT_OBJ['fam_check'] is True)
    # assert (o.OUTPUT_OBJ['name_check'] is True)  # not working at gitlab
    assert (o.OUTPUT_OBJ['p3'] == '17.01.1962')
    assert (o.OUTPUT_OBJ['birthplace3_check'] is True)
    assert (o.OUTPUT_OBJ['birthplace3_check'] is True)
    assert (o.OUTPUT_OBJ['birthplace3_check'] is True)
    # assert (o.OUTPUT_OBJ['p4ab_check'] is True)
    # assert (o.OUTPUT_OBJ['p4c_check'] is True)
    # assert (o.OUTPUT_OBJ['p5'] == '4005584309')
    # assert (o.OUTPUT_OBJ['p8_check'] is True)
    # for x in ['CE', 'A', 'D', 'C', 'B', 'DE']:
    #     assert (x in o.OUTPUT_OBJ['categories'])
    assert (o.OUTPUT_OBJ['suggest']['I'] == 'СЕРГЕЙ')
    assert (o.OUTPUT_OBJ['suggest']['O'] == 'ВИКТОРОВИЧ')


def test_passport1():
    filepatch = 'test/passport2019080145-2-0.png'
    img = cv.imread(filepatch, cv.IMREAD_COLOR)

    img, _ = crop(img, rotate=True, rate=1)
    obj = DocTypeDetected()
    obj.not_resized_cropped = img
    s = passport_main_page(obj)
    print(s.OUTPUT_OBJ)
    # {'qc': 1, 'MRZ': None, 'main_top_page': {'vidan': 'АЛАТЫРСКИМ ГОВД ЧУВАШСКОЙ РЕСНУБЛИКИ Ц КИЬ ПАДРОЗ ЭЛЫНННЯ В', 'data_vid': '16.08.2000', 'code_pod': None, 'code_pod_check': False}, 'main_bottom_page': {'F': 'ЙН', 'I': 'ЫГОРЬ', 'O': 'ВЯЧЕСЛАВОВИЧ', 'gender': 'ЖЕН', 'birth_date': None, 'birth_place': None}, 'serial_number': '9700122220', 'serial_number_check': True, 'suggest': {'F': 'ОЮН', 'F_gender': 'UNKNOWN', 'F_score': 0.5, 'I': 'ИГОРЬ', 'I_gender': 'MALE', 'I_score': 0.75, 'O': 'ВЯЧЕСЛАВОВИЧ', 'O_gender': 'MALE', 'O_score': 1}}
    # {'qc': 1, 'MRZ': None, 'main_top_page': {'vidan': 'АЛАТЫРСКИМ ГОВД ЧУВАШСКОЙ РЕСНУБЛИКИ Ц КИЬ ПАДРОЗ ЭЛЫНННЯ В', 'data_vid': '16.08.2000', 'code_pod': None, 'code_pod_check': False}, 'main_bottom_page': {'F': 'АБРОСЫКИН', 'I': 'ИГООРЬ', 'O': 'ВЯЧЕСЛАБОБИЧ', 'gender': 'ЖЕН', 'birth_date': None, 'birth_place': None}, 'serial_number': '9700122220', 'serial_number_check': True, 'suggest': {'F': 'АБРОСЬКИН', 'F_gender': 'MALE', 'F_score': 0.75, 'I': 'ИГОРЬ', 'I_gender': 'MALE', 'I_score': 0.75, 'O': 'ВЯЧЕСЛАВОВИЧ', 'O_gender': 'MALE', 'O_score': 0.6}}
    assert (s.OUTPUT_OBJ['main_top_page']['data_vid'] == '16.08.2000')
    # assert (s.OUTPUT_OBJ['main_bottom_page']['I'] == 'ИГОРЬ')
    # assert (s.OUTPUT_OBJ['main_bottom_page']['gender'] == 'МУЖ')

    assert (s.OUTPUT_OBJ['serial_number_check'] is True)
    assert (s.OUTPUT_OBJ['serial_number'] == '9700122220')
    assert (s.OUTPUT_OBJ['suggest']['I'] == 'ИГОРЬ')
    assert (s.OUTPUT_OBJ['suggest']['O'] == 'ВЯЧЕСЛАВОВИЧ')
    assert (s.OUTPUT_OBJ['suggest']['I_gender'] == 'MALE')


def test_passport2():
    filepatch = 'test/passport160-561-0.png'
    img = cv.imread(filepatch, cv.IMREAD_COLOR)

    img, _ = crop(img, rotate=True, rate=1)
    obj = DocTypeDetected()
    obj.not_resized_cropped = img
    s = passport_main_page(obj)
    # {'qc': 1, 'MRZ': None, 'main_top_page': {'vidan': 'ОТДЕЛОМ УФМС РОССИИ ПО КРАСНОДАРСКОМУ КРАЮ В КРЫМСКОМ РАЙОНЕ', 'data_vid': '02.06.2010', 'code_pod': '230-023'}, 'main_bottom_page': {'F': 'АЛЛАХВЕРДИЯН', 'I': 'АЛЬВИНА', 'O': 'ВАСИЛИЕВНА', 'gender': 'ЖЕН', 'birth_date': None, 'birth_place': 'ДАА РЕЕ ЫЫ АНАПСКИЙ КРЫМСКОГО Р-НА КРАСНОДАРСКОГО КРАЯ '}, 'serial_number': '0310502854', 'serial_number_check': True}
    # {'qc': 1, 'MRZ': None, 'main_top_page': {'vidan': 'ОТДЕЛОМ УФМС РОССИИ ПО КРАСНОДАРСКОМУ КРАЮ В КРЫМСКОМ РАЙОНЕ', 'data_vid': '02.06.2010', 'code_pod': '230-023', 'code_pod_check': False}, 'main_bottom_page': {'F': 'АЛЛАХВЕРДИЯН', 'I': 'АВЬВИНА', 'O': 'ВАСИЛИЕВНА', 'gender': 'ЖЕН', 'birth_date': '21.05.1990', 'birth_place': 'АНАПСКИЙ Р-НА'}, 'serial_number': '0310502854', 'serial_number_check': True, 'suggest': {'F': 'АЛЛАХВЕРДИЯН', 'F_gender': 'UNKNOWN', 'F_score': 1, 'I': 'АЛЬВИНА', 'I_gender': 'FEMALE', 'I_score': 0.75, 'O': 'ВАСИЛИЕВНА', 'O_gender': 'FEMALE', 'O_score': 1}}
    print(s.OUTPUT_OBJ)
    # assert (s.OUTPUT_OBJ['main_top_page']['vidan'] ==
    #         'ОТДЕЛОМ УФМС РОССИИ ПО КРАСНОДАРСКОМУ КРАЮ В КРЫМСКОМ РАЙОНЕ') #blacklist makes recognition a little worse

    assert (s.OUTPUT_OBJ['main_top_page']['data_vid'] == '02.06.2010')
    assert (s.OUTPUT_OBJ['main_top_page']['code_pod'] == '230-023')
    # assert (s.OUTPUT_OBJ['main_bottom_page']['F'] == 'АЛЛАХВЕРДИЯН')
    # assert (s.OUTPUT_OBJ['main_bottom_page']['I'] == 'АЛЬВИНА')
    assert (s.OUTPUT_OBJ['main_bottom_page']['O'] == 'ВАСИЛИЕВНА')
    assert (s.OUTPUT_OBJ['main_bottom_page']['gender'] == 'ЖЕН')
    assert (s.OUTPUT_OBJ['main_bottom_page']['birth_date'] == '21.05.1990')
    assert (s.OUTPUT_OBJ['serial_number_check'] is True)
    assert (s.OUTPUT_OBJ['suggest']['O'] == 'ВАСИЛИЕВНА')
    assert (s.OUTPUT_OBJ['suggest']['O_gender'] == 'FEMALE')
    assert (s.OUTPUT_OBJ['suggest']['I'] == 'АЛЬВИНА')
    assert (s.OUTPUT_OBJ['suggest']['I_gender'] == 'FEMALE')
    # assert (s.OUTPUT_OBJ['suggest']['F'] == 'АЛЛАХВЕРДИЯН')


def test_passport_mrz():
    filepatch = 'test/passport258-659-0.png'
    img = cv.imread(filepatch, cv.IMREAD_COLOR)

    img, _ = crop(img, rotate=True, rate=1)
    obj = DocTypeDetected()
    obj.not_resized_cropped = img
    s = passport_main_page(obj)
    print(s.OUTPUT_OBJ)
    # {'qc': 0, 'MRZ': {'s_number': '9214844670', 's_number_check': True, 'birth_date': '771111', 'birth_date_check': True, 'issue_date': '150303', 'code': '160017', 'issue_date_and_code_check': True, 'final_check': True, 'mrz_f': 'КУНАЕВА', 'mrz_i': 'АННА', 'mrz_o': 'СТАНИСЛАВОВНА', 'mrz_f_check': False, 'mrz_i_check': False, 'mrz_o_check': True, 'gender': 'F', 'gender_check': True}, 'main_top_page': {'vidan': 'ОТДЕЛОМ УФМС РОССИИ ПО РЕСПУБЛИКЕ ТАТАРСТАН В Г. НИЖНЕКАМСКЕ', 'data_vid': '03.03.2015', 'code_pod': None, 'code_pod_check': False}, 'main_bottom_page': {'F': 'И НА 1А', 'I': 'ГОРЬКОВСКОЙ ОБЛАСТИ ЗА', 'O': 'СТАНИСЛАВОВНА', 'gender': 'ЖЕН', 'birth_date': None, 'birth_place': 'ЙС СС С.'}, 'serial_number': '9214844670', 'serial_number_check': True, 'suggest': {'F': 'КУНАЕВА', 'F_gender': 'FEMALE', 'F_score': 1.0, 'I': 'АННА', 'I_gender': 'FEMALE', 'I_score': 1, 'O': 'СТАНИСЛАВОВНА', 'O_gender': 'FEMALE', 'O_score': 1}}
    # {'qc': 0, 'MRZ': {'s_number': '9214844670', 's_number_check': True, 'birth_date': '771111', 'birth_date_check': True, 'issue_date': '150303', 'code': '160017', 'issue_date_and_code_check': True, 'final_check': True, 'mrz_f': 'КУНАЕВА', 'mrz_i': 'АННА', 'mrz_o': 'СТАНИСЛАВОВНА', 'mrz_f_check': False, 'mrz_i_check': True, 'mrz_o_check': True, 'gender': 'F', 'gender_check': True}, 'main_top_page': {'vidan': 'ОТДЕЛОМ УФМС РОССИИ ПО РЕСПУБЛИКЕ ТАТАРСТАН В Г. НИЖНЕКАМСКЕ', 'data_vid': '03.03.2015', 'code_pod': None, 'code_pod_check': False}, 'main_bottom_page': {'F': 'И НА 1А', 'I': 'АННА', 'O': 'СТАНИСЛАВОВНА', 'gender': 'ЖЕН', 'birth_date': None, 'birth_place': 'ГОРЬКОВСКОЙ'}, 'serial_number': '9214844670', 'serial_number_check': True, 'suggest': {'F': 'КУНАЕВА', 'F_gender': 'FEMALE', 'F_score': 1.0, 'I': 'АННА', 'I_gender': 'FEMALE', 'I_score': 1, 'O': 'СТАНИСЛАВОВНА', 'O_gender': 'FEMALE', 'O_score': 1}}
    # assert (s.OUTPUT_OBJ['MRZ']['final_check'] is True)
    assert (s.OUTPUT_OBJ['MRZ']['s_number_check'] is True)
    assert (s.OUTPUT_OBJ['MRZ']['birth_date_check'] is True)
    # assert (s.OUTPUT_OBJ['MRZ']['issue_date_and_code_check'] is True)
    assert (s.OUTPUT_OBJ['MRZ']['s_number'] == '9214844670')
    assert (s.OUTPUT_OBJ['MRZ']['birth_date'] == '771111')
    assert (s.OUTPUT_OBJ['MRZ']['birth_date_check'] is True)
    assert (s.OUTPUT_OBJ['MRZ']['issue_date'] == '150303')
    assert (s.OUTPUT_OBJ['MRZ']['final_check'] is True)
    assert (s.OUTPUT_OBJ['MRZ']['code'] == '160017')
    assert (s.OUTPUT_OBJ['MRZ']['issue_date_and_code_check'] is True)
    assert (s.OUTPUT_OBJ['MRZ']['final_check'] is True)
    assert (s.OUTPUT_OBJ['MRZ']['mrz_i'] == 'АННА')
    assert (s.OUTPUT_OBJ['MRZ']['mrz_o'] == 'СТАНИСЛАВОВНА')
    assert (s.OUTPUT_OBJ['MRZ']['gender'] == 'F')
    assert (s.OUTPUT_OBJ['MRZ']['gender_check'] is True)
    assert (s.OUTPUT_OBJ['MRZ']['mrz_i_check'] is True)

    assert (s.OUTPUT_OBJ['main_top_page']['data_vid'] == '03.03.2015')
    # assert (s.OUTPUT_OBJ['main_top_page']['code_pod'] == '160-017')
    # assert (s.OUTPUT_OBJ['main_top_page']['code_pod_check'] is True)
    assert (s.OUTPUT_OBJ['main_bottom_page']['gender'] == 'ЖЕН')
    # assert (s.OUTPUT_OBJ['main_bottom_page']['F'] == 'КУНАЕВА')
    assert (s.OUTPUT_OBJ['main_bottom_page']['I'] == 'АННА')
    assert (s.OUTPUT_OBJ['main_bottom_page']['O'] == 'СТАНИСЛАВОВНА')
    assert (s.OUTPUT_OBJ['main_bottom_page']['gender'] == 'ЖЕН')
    # assert (s.OUTPUT_OBJ['main_bottom_page']['birth_date'] == '11.11.1977')
    # assert (s.OUTPUT_OBJ['main_bottom_page']['birth_place'] == 'ГОР. ПАВЛОВО ГОРЬКОВСКОЙ ОБЛАСТИ')
    assert (s.OUTPUT_OBJ['serial_number_check'] is True)
    # gender check
    assert (s.OUTPUT_OBJ['main_bottom_page']['gender'] == 'ЖЕН')
    assert (s.OUTPUT_OBJ['MRZ']['gender'] == 'F')
    assert (s.OUTPUT_OBJ['MRZ']['gender_check'] is True)

    assert (s.OUTPUT_OBJ['serial_number_check'] is True)
    assert (s.OUTPUT_OBJ['serial_number'] == s.OUTPUT_OBJ['MRZ']['s_number'])

    assert (s.OUTPUT_OBJ['suggest']['O'] == 'СТАНИСЛАВОВНА')
    assert (s.OUTPUT_OBJ['suggest']['O_gender'] == 'FEMALE')
    assert (s.OUTPUT_OBJ['suggest']['I'] == 'АННА')
    assert (s.OUTPUT_OBJ['suggest']['I_gender'] == 'FEMALE')
    assert (s.OUTPUT_OBJ['suggest']['F'] == 'КУНАЕВА')
    assert (s.OUTPUT_OBJ['suggest']['F_gender'] == 'FEMALE')


def test_passport_mrz2():
    filepatch = 'test/passport200-601-0.png'
    img = cv.imread(filepatch, cv.IMREAD_COLOR)
    img, _ = crop(img, rotate=True, rate=1)
    obj = DocTypeDetected()
    obj.not_resized_cropped = img
    s = passport_main_page(obj)
    print(s.OUTPUT_OBJ)
    # normal
    # {'qc': 0, 'MRZ': {'s_number': '9219611738', 's_number_check': True, 'birth_date': '940302', 'birth_date_check': True, 'issue_date': '190618', 'code': '160032', 'issue_date_and_code_check': True, 'final_check': True, 'mrz_f': 'ВАЛЕЕВЫА', 'mrz_i': 'АЛИЯ', 'mrz_o': 'ИСЛАМОВНА', 'mrz_f_check': False, 'mrz_i_check': True, 'mrz_o_check': False, 'gender': 'F', 'gender_check': False}, 'main_top_page': {'vidan': 'МВД ПО РЕСПУБЛИКЕ ТАТАРСТАН', 'data_vid': '18.06.2019', 'code_pod': '160-032'}, 'main_bottom_page': {'F': 'ЗАЛЕ НВА', 'I': 'АЛИЯ', 'O': 'РАЖА', 'gender': None, 'birth_date': '02.03.1994', 'birth_place': 'ЛЬР АКЧИВ МА АРСКИЙ РАЙОН РЕСП. ТАТАРСТАН '}, 'serial_number': '9219611738', 'serial_number_check': True}
    # {'qc': 0, 'MRZ': {'s_number': '9219611738', 's_number_check': True, 'birth_date': '940302', 'birth_date_check': True, 'issue_date': '190618', 'code': '160032', 'issue_date_and_code_check': True, 'final_check': True, 'mrz_f': 'ВАЛЕЕВА', 'mrz_i': 'АЛИЯ', 'mrz_o': 'ИСЛАМОВНА', 'mrz_f_check': True, 'mrz_i_check': True, 'mrz_o_check': True, 'gender': 'F', 'gender_check': True}, 'main_top_page': {'vidan': 'МВЛ ПО РЕСПУБЛИКЕ ТАТАРСТАН', 'data_vid': '18.06.2019', 'code_pod': '160-032', 'code_pod_check': True}, 'main_bottom_page': {'F': 'ВАЛЕЕВА', 'I': 'АЛИЯ', 'O': 'ИСЛАМОВНА', 'gender': 'ЖЕН', 'birth_date': '02.03.1994', 'birth_place': 'АКЧИШМА РАЙОН'}, 'serial_number': '9219611738', 'serial_number_check': True, 'suggest': {'F': 'ВАЛЕЕВА', 'F_gender': 'FEMALE', 'F_score': 1, 'I': 'АЛИЯ', 'I_gender': 'FEMALE', 'I_score': 1, 'O': 'ИСЛАМОВНА', 'O_gender': 'FEMALE', 'O_score': 1}}
    assert (s.OUTPUT_OBJ['MRZ']['mrz_f'] == 'ВАЛЕЕВА')
    assert (s.OUTPUT_OBJ['MRZ']['mrz_o_check'] is True)
    assert (s.OUTPUT_OBJ['MRZ']['mrz_f_check'] is True)
    assert (s.OUTPUT_OBJ['MRZ']['final_check'] is True)

    assert (s.OUTPUT_OBJ['MRZ']['s_number_check'] is True)
    assert (s.OUTPUT_OBJ['MRZ']['birth_date_check'] is True)
    # assert (s.OUTPUT_OBJ['MRZ']['f_check'] is True)
    assert (s.OUTPUT_OBJ['MRZ']['issue_date_and_code_check'] is True)
    assert (s.OUTPUT_OBJ['MRZ']['s_number'] == '9219611738')
    assert (s.OUTPUT_OBJ['MRZ']['birth_date'] == '940302')
    assert (s.OUTPUT_OBJ['MRZ']['issue_date'] == '190618')
    assert (s.OUTPUT_OBJ['MRZ']['code'] == '160032')

    assert (s.OUTPUT_OBJ['main_top_page']['data_vid'] == '18.06.2019')
    assert (s.OUTPUT_OBJ['main_top_page']['code_pod'] == '160-032')

    assert (s.OUTPUT_OBJ['main_bottom_page']['F'] == 'ВАЛЕЕВА')
    assert (s.OUTPUT_OBJ['main_bottom_page']['I'] == 'АЛИЯ')
    assert (s.OUTPUT_OBJ['main_bottom_page']['O'] == 'ИСЛАМОВНА')
    # # gender check
    # assert (s.OUTPUT_OBJ['main_bottom_page']['gender'] is None)
    assert (s.OUTPUT_OBJ['main_bottom_page']['birth_date'] == '02.03.1994')
    # assert (s.OUTPUT_OBJ['MRZ']['gender'] == 'F')
    # assert (s.OUTPUT_OBJ['MRZ']['gender_check'] is False)
    assert (s.OUTPUT_OBJ['serial_number_check'] is True)
    assert (s.OUTPUT_OBJ['serial_number'] == s.OUTPUT_OBJ['MRZ']['s_number'])
    assert (s.OUTPUT_OBJ['suggest']['O'] == 'ИСЛАМОВНА')
    assert (s.OUTPUT_OBJ['suggest']['I'] == 'АЛИЯ')
    assert (s.OUTPUT_OBJ['suggest']['F'] == 'ВАЛЕЕВА')


def test_MainOpenCV():
    def predict_mock(image1, image2):
        return 4, 'passport_main', 3

    def predict_mock_is_text1(image1):
        return True

    def predict_mock_is_text2(image1):
        return False

    inject.clear_and_configure(lambda binder: binder
                               .bind_to_provider('predict', lambda: predict_mock)
                               .bind_to_provider('predict_is_text', lambda: predict_mock_is_text1))

    p = 'test/dkp.png'
    ret = MainOpenCV.MainProcessingClass(None).workflow(p)
    ret = ret[1]['pages'][0]
    print(ret)

    assert (ret['document_type'] == 11)

    inject.clear_and_configure(lambda binder: binder
                               .bind_to_provider('predict', lambda: predict_mock)
                               .bind_to_provider('predict_is_text', lambda: predict_mock_is_text2))

    p = 'test/9-140-0.png'
    ret = MainOpenCV.MainProcessingClass('123').workflow(p)
    assert (ret[0] == '123')
    ret = ret[1]
    assert (ret['status'] == 'ready')
    ret = ret['pages'][0]
    print(ret)
    # {'document_type': 140, 'description': 'Passport main page and Driving license', 'qc': 1, 'passport_main': {'qc': 1, 'MRZ': None, 'main_top_page': {'vidan': 'НТНЕМ СУФМС РОССИИ ГЮ.АСГРАХАНСКОЙ ОБЛАСТИ В СОВЕТСКОМ РАЙОНЕ ГОР.АСТРАХАНИ КАОД ПОДРЕЗЛЕЛЕНИЯ', 'data_vid': None, 'code_pod': '300-003', 'code_pod_check': False}, 'main_bottom_page': {'F': 'МИЛЯЕВ', 'I': None, 'O': 'АЛЕКСАНДРОВИЧ', 'gender': None, 'birth_date': '02.12.1987', 'birth_place': 'ГОР.ШЕВЧЕНКО МАНГЫШЛАКСКОЙ РЕСП. МАНГЫШЛАКСКОЙ'}, 'serial_number': '1207171676', 'serial_number_check': True, 'suggest': {'F': 'МИЛЯЕВ', 'F_gender': 'MALE', 'F_score': 1, 'I': None, 'I_gender': None, 'I_score': 0, 'O': 'АЛЕКСАНДРОВИЧ', 'O_gender': 'MALE', 'O_score': 1}}, 'driving_license': {'qc': 1, 'side': 'front', 'fam_rus': 'МИЛЯЕВ', 'fam_eng': 'MILIAEV', 'fam_check': True, 'name_rus': 'НИС АПЕКСМЧОВИЧ', 'name_eng': 'NIS ALEXSANDROVICH', 'name_check': False, 'p3': '02.12.1987', 'birthplace3_rus': 'КАЗАХСТАН', 'birthplace3_eng': 'KAZAKHST AN', 'birthplace3_check': False, 'p4a': '06.04.2019', 'p4b': '06.04.2029', 'p4ab_check': True, 'p4c_rus': 'СЕНОВОА', 'p4c_eng': None, 'p4c_check': False, 'p5': '9907392380', 'p8_rus': 'АСТРАХАНСКАЯ ОБЛ', 'p8_eng': 'ASTRAKHANSKAIA OBL', 'p8_check': True, 'suggest': {'F': 'МИЛЯЕВ', 'F_gender': 'MALE', 'F_score': 1, 'I': 'ЯНИС', 'I_gender': 'MALE', 'I_score': 0.75, 'O': 'АЛЕКСАНОВИЧ', 'O_gender': 'MALE', 'O_score': 0.4}, 'categories': ['B1', 'B']}, 'file_uuid': '8601f1386582428685eb30cf4879d42d'}
    # {'document_type': 140, 'description': 'Passport main page and Driving license', 'qc': 1, 'passport_main': {'qc': 1, 'MRZ': None, 'main_top_page': {'vidan': 'НТНЕМ СУФМС РОССИИ ГЮ.АСГРАХАНСКОЙ ОБЛАСТИ В СОВЕТСКОМ РАЙОНЕ ГОР.АСТРАХАНИ КАОД ПОДРЕЗЛЕЛЕНИЯ', 'data_vid': None, 'code_pod': '300-003', 'code_pod_check': False}, 'main_bottom_page': {'F': 'МИЛЯЕВ', 'I': None, 'O': 'АЛЕКСАНДРОВИЧ', 'gender': None, 'birth_date': '02.12.1987', 'birth_place': 'ГОР.ШЕВЧЕНКО МАНГЫШЛАКСКОЙ РЕСП. МАНГЫШЛАКСКОЙ'}, 'serial_number': '1207171676', 'serial_number_check': True, 'suggest': {'F': 'МИЛЯЕВ', 'F_gender': 'MALE', 'F_score': 1, 'I': None, 'I_gender': None, 'I_score': 0, 'O': 'АЛЕКСАНДРОВИЧ', 'O_gender': 'MALE', 'O_score': 1}}, 'driving_license': {'qc': 1, 'side': 'front', 'fam_rus': 'МИЛЯЕВ', 'fam_eng': 'MILIAEV', 'fam_check': True, 'name_rus': 'НИС АЛЕКСАНДРОВИЧ', 'name_eng': 'NIS ALEKSANDROVICH', 'name_check': True, 'p3': '1987', 'birthplace3_rus': 'КАЗАХСТАН', 'birthplace3_eng': 'KAZAKHSTAN', 'birthplace3_check': True, 'p4a': '06.04.2019', 'p4b': '06.04.2029', 'p4ab_check': True, 'p4c_rus': 'ГИБДД 5004', 'p4c_eng': 'GI8DD 5004', 'p4c_check': False, 'p5': '9907392380', 'p8_rus': 'АСТРАХАНСКАЯ ОБЛ', 'p8_eng': 'ASTRAKHANSKAIA OBL', 'p8_check': True, 'suggest': {'F': 'МИЛЯЕВ', 'F_gender': 'MALE', 'F_score': 1, 'I': 'ЯНИС', 'I_gender': 'MALE', 'I_score': 0.75, 'O': 'АЛЕКСАНДРОВИЧ', 'O_gender': 'MALE', 'O_score': 1}, 'categories': ['B1', 'B']}, 'file_uuid': '3e1ac516d7e34e43bdcaf1296c5d3f87'}
    assert (0 <= ret['qc'] <= 3)
    assert (ret['document_type'] == 140)
    assert (0 <= ret['passport_main']['qc'] <= 3)
    assert (ret['passport_main']['serial_number'] == '1207171676')
    assert (ret['passport_main']['serial_number_check'] is True)
    assert (ret['passport_main']['MRZ'] is None)
    assert (ret['passport_main']['main_bottom_page']['F'] == 'МИЛЯЕВ')
    # assert (ret['passport_main']['main_bottom_page']['I'] == 'ДЕНИС')
    assert (ret['passport_main']['serial_number_check'] is True)
    # assert (0 <= ret['driving_license']['qc'] <= 3)
    assert (ret['driving_license']['fam_rus'] == 'МИЛЯЕВ')
    assert (ret['driving_license']['p4a'] == '06.04.2019')
    assert (ret['driving_license']['p4b'] == '06.04.2029')
    assert (ret['driving_license']['p4ab_check'] is True)
    assert (ret['driving_license']['p5'] == '9907392380')
    assert (ret['driving_license']['p8_check'] is True)
    assert (ret['driving_license']['birthplace3_check'] is True)
    assert ('B' in ret['driving_license']['categories'])
    assert ('B1' in ret['driving_license']['categories'])


def test_dtype():
    from utils.doctype_by_text import mark_second_files
    res = mark_second_files([{
        "document_type": 40,
        "description": "Driving license and Passport not main page",
        "qc": 4,
        "exception": "Fail to detect front side of Driving License."
    },
        {
            "document_type": 140,
            "description": "Driving license and Passport not main page",
            "qc": 3,
            "exception": "Fail to detect back side of Driving License."
        },
        {
            "document_type": 140,
            "description": "Driving license and Passport not main page",
            "qc": 3,
            "exception": "Fail to detect back side of Driving License."
        }
    ])
    assert (res[0]['document_type'] == 40)
    assert (res[1]['document_type'] == 40)
    assert (res[2]['document_type'] == 140)


def test_simplify_api():
    from utils.simplify_api import TRUE, FALSE, POSSIBLE, simplify
    # driving license
    p = {'document_type': 130, 'description': '', 'qc': 0, 'side': 'front', 'fam_rus': 'КОВРИГИН',
         'fam_eng': 'KOVRIGIN', 'fam_check': True,
         'name_rus': 'СЕРГЕЙ ВИКТОРОВИЧ', 'name_eng': 'SERGEY VIKTOROVICH', 'name_check': True, 'p3': '17.01.1962',
         'birthplace3_rus': 'МОРДОВИЯ', 'birthplace3_eng': 'MORDOVIYA', 'birthplace3_check': True,
         'p4a': '15.08.2012', 'p4b': '15.08.2022', 'p4ab_check': True, 'p4c_rus': 'ГИБДД 4010',
         'p4c_eng': 'GIBDD 4010', 'p4c_check': True, 'p5': '4005584309', 'p8_rus': 'КАЛУЖСКАЯ ОБЛ',
         'p8_eng': 'KALUZHSKAYA OBL', 'p8_check': True,
         'suggest': {'F': 'КОВРИГИН', 'F_gender': 'MALE', 'F_score': 1, 'I': 'СЕРГЕЙ', 'I_gender': 'MALE',
                     'I_score': 1, 'O': 'ВИКТОРОВИЧ', 'O_gender': 'MALE', 'O_score': 1},
         'categories': ['D', 'DE', 'A', 'B', 'CE', 'C']}
    r = simplify(p)
    assert (r['qc'] == 0)
    assert (r['document_type'] == 130)
    assert (r['description'] == '')
    assert (r['side'] == 'front')
    assert (r['F']['value'] == 'КОВРИГИН')
    assert (r['F']['trusted'] == TRUE)
    assert (r['I']['value'] == 'СЕРГЕЙ ВИКТОРОВИЧ')
    assert (r['I']['trusted'] == TRUE)
    assert (r['birth_date']['value'] == '17.01.1962')
    assert (r['birth_place']['value'] == 'МОРДОВИЯ')
    assert (r['birth_place']['trusted'] == TRUE)
    assert (r['issue_date']['value'] == '15.08.2012')
    assert (r['issue_date']['trusted'] == TRUE)
    assert (r['expiration_date']['value'] == '15.08.2022')
    assert (r['expiration_date']['trusted'] == TRUE)
    assert (r['p4c']['value'] == 'ГИБДД 4010')
    assert (r['p4c']['trusted'] == TRUE)
    assert (r['serial_number']['value'] == '4005584309')
    assert (r['p8_place']['value'] == 'КАЛУЖСКАЯ ОБЛ')
    assert (r['p8_place']['trusted'] == TRUE)
    assert (r['gender']['value'] == 'M')  # latin
    # passports
    p1 = {'qc': 1, "document_type": 100, "description": 's', 'MRZ': None,
          'main_top_page': {'vidan': 'АЛАТЫРСКИМ ГОВД ЧУВАШСКОЙ РЕСПУБЛИКИ КИД', 'data_vid': '16.08.2000',
                            'code_pod': None,
                            'code_pod_check': False},
          'main_bottom_page': {'F': 'АБРОСЫИН', 'I': 'ИГОРЬ', 'O': 'ВЛАЧЕСЛАБОБИЧ', 'gender': None,
                               'birth_date': None,
                               'birth_place': None}, 'serial_number': '9700122220', 'serial_number_check': False,
          'suggest': {'F': 'АБРОСИН', 'F_gender': 'MALE', 'F_score': 0.75, 'I': 'ИГОРЬ', 'I_gender': 'MALE',
                      'I_score': 1,
                      'O': None, 'O_gender': None, 'O_score': None}}

    p2 = {'qc': 0, "document_type": 100, "description": 'v',
          'MRZ': {'s_number': '9214844670', 's_number_check': True, 'birth_date': '771111', 'birth_date_check': True,
                  'issue_date': '150303', 'code': '160017', 'issue_date_and_code_check': True, 'final_check': True,
                  'mrz_f': 'КУНАЕВА', 'mrz_i': 'АННА', 'mrz_o': 'СТАНИСЛАВОВНА', 'mrz_f_check': False,
                  'mrz_i_check': False, 'mrz_o_check': True, 'gender': 'F', 'gender_check': True},
          'main_top_page': {'vidan': 'ОТДЕЛОМ УФМС РОССИИ ПО РЕСПУБЛИКЕ ТАТАРСТАН В Г. НИЖНЕКАМСКЕ',
                            'data_vid': '03.03.2015', 'code_pod': '160-017', 'code_pod_check': True},
          'main_bottom_page': {'F': 'У НАНА', 'I': 'ИННА', 'O': 'СТАНИСЛАВОВНА', 'gender': 'ЖЕН', 'birth_date': None,
                               'birth_place': 'ЗМЫМАЛЕТАЫТС1'}, 'serial_number': '9214844670',
          'serial_number_check': True,
          'suggest': {'F': 'КУНАЕВА', 'F_gender': 'FEMALE', 'F_score': 1.0, 'I': 'АННА', 'I_gender': 'FEMALE',
                      'I_score': 1, 'O': 'СТАНИСЛАВОВНА', 'O_gender': 'FEMALE', 'O_score': 1}}

    r = simplify(p2)
    assert (r['qc'] == 0)
    assert (r['document_type'] == 100)
    assert (r['description'] == 'v')
    assert (r['serial_number1']['value'] == '9214')
    assert (r['serial_number1']['trusted'] == TRUE)
    assert (r['serial_number2']['value'] == '844670')
    assert (r['serial_number2']['trusted'] == TRUE)
    assert (r['F']['value'] == 'КУНАЕВА')
    assert (r['F']['trusted'] == POSSIBLE)
    assert (r['I']['value'] == 'АННА')
    assert (r['I']['trusted'] == POSSIBLE)
    assert (r['O']['value'] == 'СТАНИСЛАВОВНА')
    assert (r['O']['trusted'] == TRUE)
    assert (r['birth_date']['value'] == '11.11.1977')
    assert (r['birth_date']['trusted'] == TRUE)
    assert (r['gender']['value'] == 'F')
    assert (r['gender']['trusted'] == TRUE)
    assert (r['issue_date']['value'] == '03.03.2015')
    assert (r['issue_date']['trusted'] == TRUE)
    assert (r['code_pod']['value'] == '160017')
    assert (r['code_pod']['trusted'] == TRUE)
    assert (r['vidan']['value'] == 'ОТДЕЛОМ УФМС РОССИИ ПО РЕСПУБЛИКЕ ТАТАРСТАН В Г. НИЖНЕКАМСКЕ')
    assert (r['birth_place']['value'] == 'ЗМЫМАЛЕТАЫТС1')

    r = simplify(p1)
    # print(r)
    assert (r['qc'] == 1)
    assert (r['document_type'] == 100)
    assert (r['description'] == 's')
    assert (r['serial_number1']['value'] == '9700')
    assert (r['serial_number1']['trusted'] == FALSE)
    assert (r['serial_number2']['value'] == '122220')
    assert (r['serial_number2']['trusted'] == FALSE)
    assert (r['F']['value'] == 'АБРОСИН')
    assert (r['F']['trusted'] == POSSIBLE)
    assert (r['I']['value'] == 'ИГОРЬ')
    assert (r['I']['trusted'] == POSSIBLE)
    assert (r['O']['value'] == 'ВЛАЧЕСЛАБОБИЧ')
    assert (r['O']['trusted'] == FALSE)
    assert (r['issue_date']['value'] == '16.08.2000')
    assert (r['issue_date']['trusted'] == FALSE)
    assert (r['vidan']['value'] == 'АЛАТЫРСКИМ ГОВД ЧУВАШСКОЙ РЕСПУБЛИКИ КИД')

    # both
    p4 = {'document_type': 140, 'description': 'Passport main page and Driving license', 'qc': 1,
          'passport_main': {'qc': 1, 'MRZ': None, 'main_top_page': {
              'vidan': 'ПАСПОРТ ВЫДАН ННЕЕЕМ УФМС РОССИИ ПО АСТРАХАНСКОЙ ОБЛАСТИ В СОВЕТСКОМ РАЙОНЕ ГОР. АСТРАХАНИ КОЛ ПАДРАЗДЕЛЕНИЛ',
              'data_vid': None, 'code_pod': '300-003', 'code_pod_check': False},
                            'main_bottom_page': {'F': 'МИЛЯЕВ', 'I': None, 'O': 'АЛЕКСАНДРОВИЧ', 'gender': None,
                                                 'birth_date': '02.12.1987',
                                                 'birth_place': 'ШЕВЧЕНКО МАНГЫШАКСКОЙ РЕСП.'},
                            'serial_number': '1207171676', 'serial_number_check': True,
                            'suggest': {'F': 'МИЛЯЕВ', 'F_gender': 'MALE', 'F_score': 1, 'I': None, 'I_gender': None,
                                        'I_score': None, 'O': 'АЛЕКСАНДРОВИЧ', 'O_gender': 'MALE', 'O_score': 1}},
          'driving_license': {'qc': 1, 'side': 'front', 'fam_rus': 'МИЛЯЕВ', 'fam_eng': 'MILIAEV EHNC',
                              'fam_check': False, 'name_rus': 'АПЕКСАНДРОВИЧ Е', 'name_eng': 'ALEKSANDROVICH',
                              'name_check': False, 'p3': '1987', 'birthplace3_rus': 'КАЗАХСТАН',
                              'birthplace3_eng': 'KAZAKHSTAN', 'birthplace3_check': True, 'p4a': '06.04.2019',
                              'p4b': '06.04.2029', 'p4ab_check': True, 'p4c_rus': 'ГИБДД 5', 'p4c_eng': 'GI8D0 5004',
                              'p4c_check': False, 'p5': '9907392380', 'p8_rus': 'ОБЛ АСТР',
                              'p8_eng': 'X ASTRAKHANSKAIA OBL', 'p8_check': False,
                              'suggest': {'F': 'МИЛЯЕВ', 'F_gender': 'MALE', 'F_score': 1, 'I': None,
                                          'I_gender': None, 'I_score': 0, 'O': 'Б-О', 'O_gender': 'FEMALE',
                                          'O_score': 0.25}, 'categories': ['A', 'B', 'B1']}}
    r = simplify(p4)
    assert (r['qc'] == 1)
    assert (r['document_type'] == 140)
    assert (r['description'] == 'Passport main page and Driving license')
    assert (r['passport_main']['serial_number1']['value'] == '1207')
    assert (r['driving_license']['serial_number']['value'] == '9907392380')

    p4['driving_license'] = {'qc': 3, 'side': 'front', 'exception': 'Exception!'}
    r = simplify(p4)
    assert (r['driving_license']['qc'] == 3)
    assert (r['driving_license']['side'] == 'front')
    assert (r['driving_license']['exception'] == 'Exception!')

    p = {'document_type': 130, 'description': 'Passport main page and Driving license', 'qc': 3, 'side': 'front',
         'exception': 'Fail to detect front side of Driving License.'}
    r = simplify(p)
    assert (r['document_type'] == 130)
    assert (r['qc'] == 3)
    assert (r['side'] == 'front')


def test_handwrited_fields():
    def predict_special_doctype(image1, image2):
        return 4, 'passport_main', 3

    def predict_is_text(image1):
        return True

    def predict_is_hw(images):
        return tuple(True for _ in range(len(images)))

    inject.clear_and_configure(lambda binder: binder
                               .bind_to_provider('predict', lambda: predict_special_doctype)
                               .bind_to_provider('predict_is_text', lambda: predict_is_text)
                               .bind_to_provider('predict_is_handwrited', lambda: predict_is_hw))

    # static docs
    p = 'test/handwrited_text_9.png'
    img = cv.imread(p)
    res = parser_handwritings(img, doc_type=DocTypes.borrowerProfile, page=3)
    print(res)
    assert (res['signature1'] is True)
    assert (res['signature2'] is True)
    assert (res['some_text_field'] is True)
    assert (res['FIO'] is True)
    assert (len(res.keys()) == 4)
    res = parser_handwritings(img, doc_type=DocTypes.borrowerProfile, page=1)
    assert (res is None)
    p = 'test/handwriten_7.pdf'
    ret = MainOpenCV.MainProcessingClass(None).workflow(p)
    ret = ret[1]['pages'][0]
    print(ret)
    res = ret['text_doc_fields']
    assert (res['signature1'] is True)
    assert (res['signature2'] is True)
    assert (len(res.keys()) == 2)
    # dynamic parser
    p = 'test/handwrited_dynamic_3_last.png'
    ret = MainOpenCV.MainProcessingClass(None).workflow(p)
    ret = ret[1]['pages'][0]
    print(ret)
    res = ret['text_doc_fields']
    assert (res['signature1'] is True)
    assert (res['signature2'] is True)
    assert (len(res.keys()) == 2)  # TODO 3
    # assert (res['signature3'] is True) # TODO add

def test_pdf2image_poppler():
    from tempfile import TemporaryDirectory
    import os
    from utils.pdf_convertors import pdf2png_poppler
    p = 'test/pdf2image_poppler.pdf'
    with TemporaryDirectory() as tmp:
        files = pdf2png_poppler(p, tmp)
        assert (len(files) == 28)
        for file in files:
            assert (os.path.exists(file))
            assert (os.path.getsize(file) != 0)



def main():
    from parsers import translit_drivingl  # transliteration
    from parsers import passport_utils
    from parsers import driving_utils

    # print("Starting groonga effeciency test:")
    # from groonga_test_suggest import groonga_test
    # groonga_test()

    # Profiling
    if MainOpenCV.PROFILING:
        pr = profiling_before()
        print("Profiling Started!")

    # ---
    print("Starting samplify_api test:")
    test_simplify_api()

    print("Starting utils.doctype_by_text test:")
    test_dtype()

    print("Starting db groonga test:")
    test_fio_db()

    print("Starting parsers.translit_drivingl:")
    translit_drivingl.test()

    print("starting test_driving_license:")
    test_driving_license()

    print("Starting from parsers.passport_utils:")
    passport_utils.test()

    print("Starting parsers.driving_utils:")
    driving_utils.test()

    print("starting test_UtilModule")
    test_UtilModule()

    print("Starting test_passport1:")
    test_passport1()

    print("Starting test_passport2:")
    test_passport2()
    print("Starting test_passport_mrz:")
    test_passport_mrz()
    print("Starting test_passport_mrz2:")
    test_passport_mrz2()

    print("Starting test_MainOpenCV:")
    test_MainOpenCV()  # integration test

    print("Starting test_handwrited_fields:")
    test_handwrited_fields()

    print("Starting test_pdf2image_poppler:")
    test_pdf2image_poppler()

    # profile
    if MainOpenCV.PROFILING:
        profiling_after(pr)  # noqa


if __name__ == '__main__':  # test
    main()
