import os
from groonga import FIOChecker
import csv
from parsers.ocr import rus_alph
import random


def groonga_test():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    p_name = curr_dir + '/fio_csv/surname_ddata.csv'
    pop = []
    indexes = []
    with open(p_name, 'r') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
        p = -1
        for i, row in enumerate(reader):
            pp = int(row[2])
            w = row[0]
            pop.append((pp, w))
            if pp != p:
                indexes.append((i, pp))
                p = pp

    def get_random_word(seed) -> str:
        random.seed(seed)
        r_index = random.randint(0, pop[0][0])
        distance = pop[0][0]  # largest
        ind = 0
        p = 0
        for x in indexes:  # x[0] - i, x[1] - popularity
            cdist = abs(x[1] - r_index)
            if cdist < distance:
                distance = x[1] - r_index
                ind = x[0]
                p = x[0]
        l = len([x for x in pop if x[0] == p])
        return pop[ind + random.randint(0, l)][1]

    # print(get_random_word(1))

    g = FIOChecker(2)

    t = 0
    c = 0

    for i in range(200):
        wo = get_random_word(i)
        w = list(wo)
        ra = random.randint(0, len(rus_alph) - 1)
        rw = random.randint(0, len(wo) - 1)
        w[rw] = rus_alph[ra]
        w = ''.join(w)
        q = FIOChecker.wrapper_with_crop_retry(g.query_surname, w)
        # print(wo == q[0], wo, w, q[0])
        if wo == q[0]:
            t += 1
        c += 1
    accuracy = t / c
    print('accuracy:', accuracy)
    assert (accuracy > 0.88)
