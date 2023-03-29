from poyonga import Groonga
import os
import csv
from time import sleep
from pathlib import Path
import re

DISTANCE_OF_UNIMPORTANCE = 5000


class Word:
    text = None
    max_distance = 3
    # distance = DISTANCE_OF_UNIMPORTANCE


class FIOChecker:
    names_table = 'Name'
    patronymic_table = 'Patronymic'
    surname_table = 'Surname'

    __instance = None

    @staticmethod
    def __load(g, table_name, woords_gender):
        data2 = '[' + "".join(
            ['{"_key":"' + w + '","gender":' + str(gender) + '},' for w, gender in woords_gender]) + ']'
        g.call("load", table=table_name, values=data2)

    @staticmethod
    def __call(g, cmd, **kwargs):
        ret = g.call(cmd, **kwargs)
        # print(ret.status)
        # print(ret.body)
        if cmd == 'select':
            # for item in ret.items:
            #     print(item)
            return ret.items

    @staticmethod
    def _start_groonga():
        from subprocess import Popen, PIPE, call, SubprocessError
        # 'groonga -s --protocol http grb.db'
        process = Popen('groonga -sn --protocol http grb.db'.split(' '), stdout=PIPE)
        # (output, err) = process.communicate()
        # exit_code = process.wait()

    @staticmethod
    def _connect():
        g = Groonga(port=10041, protocol="http", host='0.0.0.0')
        return g

    @staticmethod
    def dec_gender(ind: int) -> str:
        if ind == 1:
            return 'MALE'
        elif ind == 2:
            return 'FEMALE'
        else:
            return 'UNKNOWN'

    @staticmethod
    def wrapper_with_crop_retry(query: callable, word: callable) -> (str, str, float) or None:
        """ retry with loser score and one of the word in sentence

        :param query:query_surname, query_patronymic or query_name
        :param word:
        :return: (word, gender, score) or None
        """
        ret = query(word)
        if ret is None:  # if fail we will try with parts
            splited = word.split()
            if len(splited) > 1:
                for i in range(len(splited), 0, -1):  # 3, 2, 1
                    ret = query(' '.join(splited[:i]), lower_score=True)
                    if ret:  # stop at first success
                        break
        return ret

    def _load_groonga(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))

        self.conn.call('table_create', name=FIOChecker.names_table, flags='TABLE_HASH_KEY', key_type='ShortText')
        self.conn.call('column_create', table=FIOChecker.names_table, name='gender', type='UInt8')
        self.conn.call('table_create', name=FIOChecker.patronymic_table, flags='TABLE_HASH_KEY', key_type='ShortText')
        self.conn.call('column_create', table=FIOChecker.patronymic_table, name='gender', type='UInt8')
        self.conn.call('table_create', name=FIOChecker.surname_table, flags='TABLE_HASH_KEY', key_type='ShortText')
        self.conn.call('column_create', table=FIOChecker.surname_table, name='gender', type='UInt8')

        # 'table_create --name Site --flags TABLE_HASH_KEY --key_type ShortText'

        def load(file_path, table_name):
            words = []
            with open(file_path, 'r') as f:
                reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
                for i, row in enumerate(reader):
                    if row[1] == 'MALE':
                        gender = 1
                    elif row[1] == 'FEMALE':
                        gender = 2
                    else:
                        gender = 0
                    words.append((row[0], gender))

                    if i % 1000 == 0:  # every 1000 records
                        self.__load(self.conn, table_name, words)

                        words = []

            self.__load(self.conn, table_name, words)

        p_name = curr_dir + '/fio_csv/name_ddata.csv'
        p_patr = curr_dir + '/fio_csv/patronymic_ddata.csv'
        p_sur = curr_dir + '/fio_csv/surname_ddata.csv'
        load(p_name, self.names_table)
        load(p_patr, self.patronymic_table)
        load(p_sur, self.surname_table)

    def __init__(self, timeout=1):
        # remove database before start
        for p in Path(".").glob("grb.db*"):
            p.unlink()

        self._start_groonga()  # init database
        sleep(timeout)
        self.conn = self._connect()
        self._load_groonga()

    @staticmethod
    def _get_appropriate(items: list, word: Word, lower_score=False):
        """

        :param items: [(_id, _key, _score, gender)
        :param word:
        :return:
        """
        items = sorted(items,
                       key=lambda x: (-x['_score'], x['_id']))  # score = desc  id = ascending # def = asc
        # [print(x) for x in items]
        l1 = [x for x in items if x['_score'] == word.max_distance + 1]  # score
        l2 = [x for x in items if x['_score'] == word.max_distance]
        l3 = [x for x in items if x['_score'] == word.max_distance - 1]

        score = items[0]['_score'] / (word.max_distance + 1)

        if l1 and l2:  # key
            if l1[0]['_id'] > l2[0]['_id'] * DISTANCE_OF_UNIMPORTANCE:
                items = l2
                score = 0.75
            elif l1[0]['_key'] == word.text:
                items = l1
                score = 1
            else:
                items = l1
                score = 0.75

        elif l2 and l3:
            if l2 and l3 and l2[0]['_id'] > l3[0]['_id'] * DISTANCE_OF_UNIMPORTANCE:
                items = l3
            else:
                items = l2
                score = items[0]['_score'] / (word.max_distance + 1)

        if lower_score:
            score -= 0.25

        return items[0]['_key'], FIOChecker.dec_gender(items[0]['gender']), score

    def _double_query(self, word1: str, word2: str, table: str) -> (str, str, float) or None:
        """
        get suggestions for word1 and word2 and select intersection. or first of word1
        :param table:
        :param word1: priority hight
        :param word2: priority low
        :return:
        """
        if table == self.patronymic_table:
            max_dist = 4
        else:
            max_dist = 3

        w1, w2 = Word(), Word()
        w1.text, w2.text = word1, word2
        w1.max_distance, w2.max_distance = max_dist, max_dist

        items1 = self.__call(self.conn, "select", table=table,
                             filter="fuzzy_search(_key, '" + word1 + "', {\"max_distance\": "+str(max_dist)+", \"max_expansion\": 5})",
                             sortby="-_score",
                             output_columns='_id, _key, _score, gender', limit=20)
        items2 = self.__call(self.conn, "select", table=table,
                             filter="fuzzy_search(_key, '" + word2 + "', {\"max_distance\": "+str(max_dist)+", \"max_expansion\": 5})",
                             sortby="-_score",
                             output_columns='_id, _key, _score, gender', limit=20)

        if items1 and len(items1) != 0 and items2 and len(items2) != 0:  # we have two words
            # optimization
            items1 = sorted(items1,
                            key=lambda x: (-x['_score'], x['_id']))  # score = desc  id = ascending # def = asc
            items2_l = [(x['_id'], x['_key'], x['_score'], x['gender']) for x in items2]  # dict to list
            items2_l = sorted(items2_l,
                              key=lambda x: (-x[2], x[0]))  # score = desc  id = ascending # def = asc
            words2 = list(zip(*items2_l))[1]  # words of items2

            res = [value for value in items1 if value['_key'] in words2]  # word in items1 equal to items2

            if res:
                # скорее всего одно из слов распознано полностью верно,
                # поэтому если результат не похож ни на что, возьмем
                # из списка если там есть полностью совподающее
                if res[0]['_key'] != word1 and res[0]['_key'] != word2:
                    appr1 = FIOChecker._get_appropriate(items1, w1)
                    appr2 = FIOChecker._get_appropriate(items2, w2)
                    if appr1[2] == 1:
                        return appr1
                    elif appr2[2] == 1:
                        return appr2

                # make score independent of priority
                score = res[0]['_score'] / (max_dist + 1)
                if res[0]['_key'] == word1 or res[0]['_key'] == word2:
                    score = 1

                return res[0]['_key'], FIOChecker.dec_gender(res[0]['gender']), score
            else:
                return items1[0]['_key'], FIOChecker.dec_gender(items1[0]['gender']), items1[0]['_score'] / 4
        elif items1 and len(items1) != 0:
            return FIOChecker._get_appropriate(items1, w1)
        elif items2 and len(items2) != 0:
            return FIOChecker._get_appropriate(items2, w2)
        else:
            return None

    # public
    def double_query_surname(self, word1: str, word2: str) -> (str, str, float) or None:
        return self._double_query(word1, word2, self.surname_table)

    # public
    def double_query_name(self, word1: str, word2: str) -> (str, str, float) or None:
        return self._double_query(word1, word2, self.names_table)

    # public
    def query_name(self, word, lower_score=False) -> (str, str, float) or None:
        """
        if exception we must restart container
        :param word:
        :return: word, gender, score
        """

        items = self.__call(self.conn, "select", table=self.names_table,
                            filter="fuzzy_search(_key, '" + word + "', {\"max_distance\": 3, \"max_expansion\": 5})",
                            sortby="-_score",
                            output_columns='_id, _key, _score, gender', limit=20)

        if items and len(items) != 0:
            a = Word()
            a.text = word
            a.max_distance = 3
            return FIOChecker._get_appropriate(items, a, lower_score)
        else:
            return None

    # public
    def query_patronymic(self, word: str, lower_score=False) -> (str, str, float) or None:
        """
        if exception we must restart container
        :param lower_score: if retry we will lower score -1
        :param word:
        :return: word, gender, score
        """
        from parsers.passport_utils import razd
        # https://fin-gazeta.ru/chto-oznachaet-ogly-v-imeni-i-otchestve/
        # тюрские
        turs_m = ('ОГЛЫ', 'ОГЛУ', 'УЛЫ', 'УУЛУ')
        # turs_f = ('КЫЗЫ', 'ГЫЗЫ')
        reres = re.search("^([А-Я]*)" + razd + "(ОГЛЫ|ОГЛУ|УЛЫ|УУЛУ|КЫЗЫ|ГЫЗЫ)$", word)
        if reres and len(reres.groups()) == 6 and reres.group(1) and reres.group(6):
            sug, gender, score = self.query_name(reres.group(1))  # TODO: add gender in query
            if reres.group(6) in turs_m:
                gender = 'MALE'
            else:
                gender = 'FEMALE'
            if lower_score:  # lower score
                score -= 0.25
            return sug + ' ' + reres.group(6), gender, score  # TODO: may be "-" sepparator
        else:
            items = self.__call(self.conn, "select", table=self.patronymic_table,
                                filter="fuzzy_search(_key, '" + word + "', {\"max_distance\": 4, \"max_expansion\": 5})",
                                sortby="-_score",
                                output_columns='_id, _key, _score, gender', limit=20)

            if items and len(items) != 0:
                a = Word()
                a.text = word
                a.max_distance = 4
                return FIOChecker._get_appropriate(items, a, lower_score)
            else:
                return None

    # public
    def query_surname(self, word, lower_score=False) -> (str, str, float) or None:
        """
        if exception we must restart container
        :param lower_score:
        :param word:
        :return: word, gender, score
        """

        items = self.__call(self.conn, "select", table=self.surname_table,
                            filter="fuzzy_search(_key, '" + word + "', {\"max_distance\": 3, \"max_expansion\": 5})",
                            sortby="-_score",
                            output_columns='_id, _key, _score, gender', limit=20)

        if items and len(items) != 0:
            a = Word()
            a.text = word
            a.max_distance = 3
            return FIOChecker._get_appropriate(items, a, lower_score)
        else:
            return None

    # public


if __name__ == '__main__':  # test
    g = FIOChecker(2)

    res = g.query_name('ВЛАДИСЛАВ')
    print(res)
    assert (res[0] == 'ВЛАДИСЛАВ' and res[1] == 'MALE' and res[2] == 1)
    exit()
    res = g.wrapper_with_crop_retry(g.query_name, 'КАТЯ КАТЕРУН')
    print(res)
    assert (res[0] == 'КАТЯ' and res[1] == 'FEMALE' and res[2] < 1)
    # exit()
    #
    # res = g.query_name('ЛЕНА')
    # print(res)
    # assert (res[0] == 'ЕЛЕНА' and res[1] == 'FEMALE')
    #
    # # print(g.query_surname("ВИЕТОРОВИЧ"))
    #
    # print(FIOChecker.wrapper_with_crop_retry(g.query_patronymic, "ВИЕТОРОВИЧ"))
    # exit(0)

    print(g.query_patronymic('ВЛАЧЕСЛАБОБИЧ'))
    # print(g.query_patronymic("ВИЕТОРОВИЧ"))
    # print(g.query_patronymic("ВИТОРОВИЧ"))
    # exit(0)
    print(g.query_name('АННА'))
    # print(g.double_query_name("АННА", "ИНА"))
    print(g.double_query_name("АААААААААНА", "АААААААААНА"))
    print(g.double_query_name("АЛЕКСЙ", "АЛЕСЕЙ"))
    # print(g.query_name('АПЕКСАНДРОВИЧ'))
    # print(g.query_name('ЛЛЛЛЛЛЛ'))
    print(g.query_patronymic('СЕРГЕЕВИЧ'))
    print(g.query_patronymic('ЮЛЯ ОГЛЫ'))
    print(g.query_patronymic('ВЛАДИМИРОВИЧ ЗЕКЦЕ МИ'))
    print(g.query_name('ЛЕНА'))
    print(g.query_name('АННА'))
    print(g.query_surname('МЮНАХОВ'))
    print(g.double_query_name("ЛЛЛЛЛЛЛЛЛЛЛ", "ЛЛЛЛЛЛЛЛЛЛЛ"))
    # # assert (g.query_name('ЛЕНА') == ('ЕЛЕНА', 2))
    print(FIOChecker.wrapper_with_crop_retry(g.query_name, 'КАТЕРУН'))
    print(FIOChecker.wrapper_with_crop_retry(g.query_name, 'КАТЯ КАТЕРУН'))
    print(FIOChecker.wrapper_with_crop_retry(g.query_patronymic, 'ВАЛЕНТИНА КЫЗЫ КАТЕРУН'))
    # assert (g.query_name('ВЛАДИСЛАВ') == ('ВЛАДИСЛАВ', 1))

# curr_dir = os.path.dirname(os.path.abspath(__file__))
    # p_name = curr_dir + '/fio_csv/patronymic_ddata.csv'
    # with open(p_name, 'r') as f:
    #     reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
    #     for row in reader:
    #         if 'ВИЕТОРОВИЧ' in row[0]:
    #             print(row)
    # # exit()