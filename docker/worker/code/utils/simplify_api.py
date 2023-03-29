VAL = "value"
VAL_SHORT = "value_short"
TRUSTED = "trusted"
# trusted
TRUE = "True"
FALSE = "False"
POSSIBLE = "Possible"

import datetime

now_year = datetime.datetime.now().year


def _mrz_date_process(date: str) -> str or None:
    """
    YYMMDD -> DD.MMY.YYYY
    """
    if date is None:
        return None
    if int(date[:2]) < (now_year - 2000):
        year = "20" + date[:2]
    else:
        year = "19" + date[:2]
    return date[-2:] + '.' + date[2:4] + '.' + year  # DDMMYYYY


def _passport(page: dict) -> dict:
    """ old page to simple page
    Переменные:
        serial_number1
        serial_number2
        F
        I
        O
        birth_date
        gender
        issue_date
        code_pod
        vidan
        birth_place
    """
    # Поля:
    # value ИЛИ value_short - обязательный
    # trusted - есть для ряда полей
    new_page = {"qc": page["qc"]}

    if "exception" in page:
        new_page["exception"] = page["exception"]

    if page['qc'] <= 2:
        # Серия паспорта ---------------
        serial_number1 = None
        serial_number1_short = None
        serial_number1_trust = FALSE

        if page["serial_number_check"] is True:
            serial_number1 = page["serial_number"][:4]
            serial_number1_trust = TRUE
        elif page["MRZ"] is not None and page["MRZ"]["s_number_check"] is True:
            serial_number1_short = page["MRZ"]["s_number"][:3]
            serial_number1_trust = TRUE
        elif page["serial_number"] is not None:
            serial_number1 = page["serial_number"][:4]

        if serial_number1 is not None:
            new_page["serial_number1"] = {VAL: serial_number1}
        if serial_number1_short is not None:
            new_page["serial_number1"] = {VAL_SHORT: serial_number1_short}
        if serial_number1 is not None or serial_number1_short is not None:
            new_page["serial_number1"][TRUSTED] = serial_number1_trust

        # Номер паспорта ---------------
        serial_number2 = None
        serial_number2_trust = FALSE

        if page["serial_number_check"] is True:
            serial_number2 = page["serial_number"][4:]
            serial_number2_trust = TRUE
        elif page["MRZ"] is not None and page["MRZ"]["s_number_check"] is True:
            serial_number2 = page["MRZ"]["s_number"][3:]
            serial_number2_trust = TRUE
        elif page["serial_number"] is not None:
            serial_number2 = page["serial_number"][4:]

        if serial_number2 is not None:
            new_page["serial_number2"] = {VAL: serial_number2, TRUSTED: serial_number2_trust}

        # Фамилия ---------------
        f = None
        f_trust = FALSE

        if page["MRZ"] is not None and page["MRZ"]["mrz_f_check"] == True:
            f = page["MRZ"]["mrz_f"]
            f_trust = TRUE
        elif page["suggest"] is not None and page["suggest"]["F"] is not None \
                and page["suggest"]["F_score"] >= 0.75:
            f = page["suggest"]["F"]
            if page["suggest"]["F_score"] == 1:
                f_trust = POSSIBLE
            elif page["suggest"]["F_score"] >= 0.75:
                f_trust = POSSIBLE
        elif page["main_bottom_page"]:
            f = page["main_bottom_page"]["F"]
        elif page["MRZ"]:
            f = page["MRZ"]["mrz_f"]

        if f is not None:
            new_page["F"] = {VAL: f, TRUSTED: f_trust}

        # Имя ---------------
        i = None
        i_trust = FALSE

        if page["MRZ"] is not None and page["MRZ"]["mrz_i_check"] is True:
            i = page["MRZ"]["mrz_i"]
            i_trust = TRUE
        elif page["suggest"] is not None and page["suggest"]["I"] is not None \
                and page["suggest"]["I_score"] >= 0.75:
            i = page["suggest"]["I"]
            if page["suggest"]["I_score"] == 1:
                i_trust = POSSIBLE
            elif page["suggest"]["I_score"] >= 0.75:
                i_trust = POSSIBLE
        elif page["main_bottom_page"]:
            i = page["main_bottom_page"]["I"]
        elif page["MRZ"]:
            i = page["MRZ"]["mrz_i"]

        if i is not None:
            new_page["I"] = {VAL: i, TRUSTED: i_trust}

        # Отчество ---------------
        o = None
        o_trust = FALSE

        if page["MRZ"] is not None and page["MRZ"]["mrz_o_check"] is True:
            o = page["MRZ"]["mrz_o"]
            o_trust = TRUE
        elif page["suggest"] is not None and page["suggest"]["O"] is not None and page["suggest"][
            "O_score"] >= 0.75:
            o = page["suggest"]["O"]
            if page["suggest"]["O_score"] == 1:
                o_trust = POSSIBLE
            elif page["suggest"]["O_score"] >= 0.75:
                o_trust = POSSIBLE
        elif page["main_bottom_page"]:
            o = page["main_bottom_page"]["O"]
        elif page["MRZ"]:
            o = page["MRZ"]["mrz_o"]

        if o is not None:
            new_page["O"] = {VAL: o, TRUSTED: o_trust}

        # Дата рождения -----------
        birth_date = None
        birth_date_trust = FALSE

        if page["MRZ"] is not None and page["MRZ"]["birth_date_check"]:
            birth_date = _mrz_date_process(page["MRZ"]["birth_date"])
            birth_date_trust = TRUE
        elif page["main_bottom_page"] is not None and page["main_bottom_page"]["birth_date"] is not None:
            birth_date = page["main_bottom_page"]["birth_date"]
        elif page["MRZ"] is not None:
            birth_date = _mrz_date_process(page["MRZ"]["birth_date"])

        if birth_date is not None:
            if len(birth_date) == 4:
                new_page["birth_date"] = {VAL_SHORT: birth_date}
            else:
                new_page["birth_date"] = {VAL: birth_date}
            new_page["birth_date"][TRUSTED] = birth_date_trust

        # Пол -----------
        gender = None
        gender_trust = FALSE
        if page["suggest"] is not None:
            genders = [page["suggest"]["F_gender"],
                       page["suggest"]["I_gender"],
                       page["suggest"]["O_gender"]]

        if page["MRZ"] is not None and page["MRZ"]["gender_check"] is True:
            gender = page["MRZ"]["gender"]
            gender_trust = TRUE
        elif gender is not None and \
                ((genders.count('MALE') == 3) or (genders.count('FEMALE') == 3)):  # not bug
            gender = 'M' if page["suggest"]["F_gender"] == 'MALE' else 'F'
            gender_trust = POSSIBLE
        elif page["main_bottom_page"] is not None:
            if page["main_bottom_page"]["gender"] == 'МУЖ':
                gender = 'M'
            if page["main_bottom_page"]["gender"] == 'ЖЕН':
                gender = 'F'

        if gender is not None:
            new_page["gender"] = {VAL: gender, TRUSTED: gender_trust}

        # Дата выдачи паспорта ----
        issue_date = None
        issue_date_trust = FALSE

        if page["MRZ"] is not None and page["MRZ"]["issue_date_and_code_check"] is True:
            issue_date = _mrz_date_process(page["MRZ"]["issue_date"])
            issue_date_trust = TRUE
        elif page["main_top_page"] is not None:
            issue_date = page["main_top_page"]["data_vid"]
        elif page["MRZ"] is not None:
            issue_date = _mrz_date_process(page["MRZ"]["issue_date"])

        if issue_date is not None:
            new_page["issue_date"] = {VAL: issue_date, TRUSTED: issue_date_trust}

        # Код подразделения ----
        code_pod = None
        code_pod_trust = FALSE

        if page["MRZ"] is not None and page["MRZ"]["issue_date_and_code_check"] is True:
            code_pod = page["MRZ"]["code"]
            code_pod_trust = TRUE
        elif page["main_top_page"] is not None:
            code_pod = page["main_top_page"]["code_pod"]
            if page["main_top_page"]["code_pod_check"] is True:  # res of comparision MRZ and main top page
                code_pod_trust = TRUE
        elif page["MRZ"] is not None:
            code_pod = page["MRZ"]["code"]

        if code_pod is not None:
            new_page["code_pod"] = {VAL: code_pod, TRUSTED: code_pod_trust}

        # Others ------
        if page["main_top_page"] is not None and page["main_top_page"]['vidan']:
            new_page["vidan"] = {VAL: page["main_top_page"]['vidan']}
        if page["main_bottom_page"] is not None:
            if page["main_bottom_page"]['birth_place']:
                new_page["birth_place"] = {VAL: page["main_bottom_page"]['birth_place']}

    return new_page


def _driving_license(page: dict) -> dict:
    """ old page to simple page
    Поля:
      value ИЛИ value_short - обязательный
      trusted - есть для ряда полей

    Переменные:
        front:
            F
            I
            birth_date
            birth_place
            issue_date
            expiration_date
            p4c
            serial_number
            p8_place
            gender
        back:
            serial_number

    """
    new_page = {"qc": page["qc"]}

    if "side" in page:
        new_page['side'] = page['side']
    if "exception" in page:
        new_page["exception"] = page["exception"]

    if page['qc'] <= 2:
        if page['side'] == 'front':
            # FIO
            # F
            if page['fam_rus']:
                t = FALSE
                v = page['fam_rus']
                if page['suggest'] is not None:  # page['fam_check'] may be short and all suggest is None
                    if page['fam_check'] and page['suggest']['F_score'] == 1:
                        t = TRUE
                        v = page['fam_rus']
                    elif page["suggest"] and page["suggest"]["F_score"] >= 0.75:
                        t = POSSIBLE
                        v = page["suggest"]['F']
                new_page["F"] = {VAL: v, TRUSTED: t}
            # I
            if page['name_rus']:
                t = FALSE
                v = page['name_rus']
                if page['suggest'] is not None:
                    if page['name_check'] and page['suggest']['I_score'] == 1:
                        t = TRUE
                        v = page['name_rus']
                    elif page["suggest"] and page["suggest"]["I_score"] >= 0.75:
                        t = POSSIBLE
                        v = page["suggest"]['I']
                new_page["I"] = {VAL: v, TRUSTED: t}
            # p3 birth date
            if page['p3']:
                if len(page['p3']) == 4:
                    new_page['birth_date'] = {VAL_SHORT: page['p3']}
                else:
                    new_page['birth_date'] = {VAL: page['p3']}
            # p3 birth place
            if page['birthplace3_rus']:
                t = TRUE if page['birthplace3_check'] else FALSE
                new_page['birth_place'] = {VAL: page['birthplace3_rus'], TRUSTED: t}
            # p4a and p4b - issue_date and expiration_date
            if page['p4a']:
                t = TRUE if page['p4ab_check'] else FALSE
                if len(page['p4a']) == 4:
                    new_page['issue_date'] = {VAL_SHORT: page['p4a'], TRUSTED: t}
                else:
                    new_page['issue_date'] = {VAL: page['p4a'], TRUSTED: t}
            if page['p4b']:
                t = TRUE if page['p4ab_check'] else FALSE
                if len(page['p4b']) == 4:
                    new_page['expiration_date'] = {VAL_SHORT: page['p4b'], TRUSTED: t}
                else:
                    new_page['expiration_date'] = {VAL: page['p4b'], TRUSTED: t}
            # p4c
            if page['p4c_rus']:
                t = TRUE if page['p4c_check'] else FALSE
                new_page['p4c'] = {VAL: page['p4c_rus'], TRUSTED: t}
            # p5 serial number
            if page['p5']:
                new_page['serial_number'] = {VAL: page['p5']}
            # p8_place
            if page['p8_rus']:
                t = TRUE if page['p8_check'] else FALSE
                new_page['p8_place'] = {VAL: page['p8_rus'], TRUSTED: t}

            # gender
            if page["suggest"]:
                if page["suggest"]["F_gender"] == 'MALE' and page["suggest"]["I_gender"] == 'MALE':
                    new_page['gender'] = {VAL: 'M'}
                elif page["suggest"]["F_gender"] == 'FEMALE' and page["suggest"]["I_gender"] == 'FEMALE':
                    new_page['gender'] = {VAL: 'F'}

            if page["categories"]:
                new_page["categories"] = {VAL: page["categories"]}

        else:  # == "back"
            if page['s_number']:
                new_page['serial_number'] = {VAL: page['s_number']}

    return new_page


def simplify(page: dict) -> dict:
    if "document_type" in page and page["document_type"] in (100, 130, 140):
        if page["qc"] < 4:  # 'Passport: main page'

            new_page = {"document_type": page["document_type"],
                        "qc": page["qc"],
                        "description": page["description"]}

            # passport front
            if page["document_type"] == 100:
                new_page.update(_passport(page))
            # driving license
            elif page["document_type"] == 130:
                new_page.update(_driving_license(page))
            # passport front and driving license
            elif page["document_type"] == 140:
                # passport
                if page['passport_main']:
                    new_page['passport_main'] = _passport(page['passport_main'])
                # driving license
                if page['driving_license']:
                    new_page['driving_license'] = _driving_license(page['driving_license'])

            return new_page
        elif page["qc"] == 4:
            return {"document_type": 0, "qc": 4}

    return page

# if __name__ == '__main__':
