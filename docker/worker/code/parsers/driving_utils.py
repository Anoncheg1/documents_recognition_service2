import re
import cv2 as cv
import os
# own
from parsers.passport_utils import find_date
from parsers.kaze_cropper import KazeCropper


def compare_dates(d_one: str, d_two: str) -> str or None:
    """ Compare
    if equal return d_one

    :param d_one:
    :param d_two:
    :return: one or two or None
    """
    if d_one is None and d_two:
        return find_date(d_two)
    elif d_one and d_two is None:
        return find_date(d_one)
    elif d_one and d_two:
        d_one = find_date(d_one)
        d_two = find_date(d_two)
        if d_one is None and d_two:
            return d_two
        elif d_one and d_two is None:
            return d_one
        elif d_one and d_two:
            if d_one < d_two:
                return d_two
            else:
                return d_one
    return None


def check_dates(d_one: str, d_two: str) -> bool:
    """ 09.10.2018 == 09.10.2028 - True
    2018 == 2028 - True
    else False

    :param d_one:
    :param d_two:
    :return: bool
    """
    if d_one is None or d_two is None:
        return False

    d_one = re.sub(r'\.', '', d_one)
    d_two = re.sub(r'\.', '', d_two)
    if len(d_one) != len(d_two):
        return False
    try:
        if (int(d_one[-2:]) + 10) == int(d_two[-2:]):
            if len(d_one) == 4:
                return True
            elif d_one[:4] == d_two[:4]:
                return True
    except ValueError:
        return False
    return False


class PravaCropper:
    """ cr_index:
     - 1,2 front
     - other = back"""
    def __init__(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        old_front = curr_dir + '/kaze_templates/Untitled9.png'  # /mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/85-383-0.png
        new_front = curr_dir + '/kaze_templates/Untitled_v2_001.png'  # /mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/23-321-0.png
        back_old = curr_dir + '/kaze_templates/obr_old_v6_v4.png'  # /mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/0/36-334-1.png
        back_new1 = curr_dir + '/kaze_templates/back_new.png'  # /mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/1/38-48-10.png
        back_new2 = curr_dir + '/kaze_templates/back_new2.png'  # /mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/2/141-542-7.png
        back_new3 = curr_dir + '/kaze_templates/back_new3.png'  # /mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/50-451-1.png
        # FAIL new obr # obr_obr2 = '/home/u2/Desktop/t3.png'  #'/home/u2/Desktop/new_obr_v1.png'  # /mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/0/76-374-1.png

        height = 674  # average prep
        width = 998  # average

        image = cv.imread(old_front, cv.IMREAD_COLOR)
        cropper1 = KazeCropper(image, width=width)

        image = cv.imread(new_front, cv.IMREAD_COLOR)
        cropper2 = KazeCropper(image, width=width)

        image = cv.imread(back_old, cv.IMREAD_COLOR)  # fastNlMeansDenoising better with colour
        image = cv.fastNlMeansDenoising(image, h=8, templateWindowSize=10)  # denoise edges
        cropper3 = KazeCropper(image, width=width)

        image = cv.imread(back_new1, cv.IMREAD_COLOR)
        image = cv.fastNlMeansDenoising(image, h=8, templateWindowSize=10)  # denoise edges
        cropper4 = KazeCropper(image, width=width)

        image = cv.imread(back_new2, cv.IMREAD_COLOR)
        image = cv.fastNlMeansDenoising(image, h=8, templateWindowSize=10)  # denoise edges
        cropper5 = KazeCropper(image, width=width)

        image = cv.imread(back_new3, cv.IMREAD_COLOR)
        image = cv.fastNlMeansDenoising(image, h=8, templateWindowSize=10)  # denoise edges
        cropper6 = KazeCropper(image, width=width)

        self.croppers = (cropper1, cropper2, cropper3, cropper4, cropper5, cropper6)  # 0-1 front 2-5 back

    def match(self, img) -> list or None:
        # img = KazeCropper.prepare(img) # too heavy for CPU
        m_counts = list()  # list of match counts

        try:
            for c in self.croppers:
                m_counts.append(c.match(img, filt=True)[0])
        except MemoryError as e:
            return None
        return m_counts

    def crop(self, img, match_counts: list):
        """ Choose appropriate cropper and crop

            :param img:
            :param match_counts:
            :return: cropped image or None and cropper_id
        """

        # print(res_cr)
        max_c = max(match_counts)
        cr_index = match_counts.index(max_c)
        suitable_cropper: KazeCropper = self.croppers[cr_index]
        if max_c > 100:
            image = suitable_cropper.crop(img, double_crop=True)
        else:
            image = suitable_cropper.crop(img, double_crop=False)
        return image, cr_index


def test():
    if not check_dates('09.10.2018', '09.10.2028'):
        print("fail1")
    if check_dates('09.1s.2018', '09.10.2028'):
        print("fail2")
    if not check_dates('2010', '2020'):
        print("fail3")


if __name__ == '__main__':  # test
    test()
