import cv2 as cv
import numpy as np
import imutils
import sys
from scipy.spatial import distance
import random
# own
from logger import logger


class KazeCropper:
    """ Detect image by template(pattern).
    Used in parsers/driving_license.py
    Used in parsers/passport_bottom_kaze.py
    1) resize original image
    2) compute keypoints of original image
    3) match() or crop() second image
    """

    def __init__(self, img_ptr, nn_match_ratio=0.71, width=None):
        """
        1)  img to gray
        2)  resize if width
        3)  compute keypoints and the descriptors
          3.1) to gray and fastDenoise
          3.2) compute keypoints and the descriptors
        :param img_ptr: cv.imread result
        :param nn_match_ratio:
        :param width:
        """
        if img_ptr is None:
            print('KazeCropper init: Could not open or find the image of pattern!')

        if len(img_ptr.shape) == 2 or img_ptr.shape[2] == 1:
            gray = img_ptr
        else:
            gray = cv.cvtColor(img_ptr, cv.COLOR_BGR2GRAY)

        if width is not None:
            gray = imutils.resize(gray, width=width)  # resized
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # clahe = cv.createCLAHE(clipLimit=0.2, tileGridSize=(30,30))
        # clahe.apply(lab_planes[0])
        self.orig_shape = gray.shape
        gray = cv.fastNlMeansDenoising(gray, h=8, templateWindowSize=10)  # denoise
        # cv.imshow('image', gray)  # show image in window
        # cv.waitKey(0)  # wait for any key indefinitely
        # cv.destroyAllWindows()  # close window q
        self.akaze = cv.AKAZE_create()

        self.kpts2, self.desc2 = self.akaze.detectAndCompute(gray, None)

        self.matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
        # self.matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMINGLUT)
        self.nn_match_ratio = nn_match_ratio  # 0.75  # Nearest neighbor matching ratio

    # private
    @classmethod
    def _filter(self, src_pts, dst_pts, height_threshold: int) -> (list, list):
        """
        Filter by constraining transfromation - we compare distances in source points and between target points

        1) find one triangle by checking 3 random points in src and dst
        2) find points by comparising distances to first triangle points and new founded

        :param src_pts: ordered lists
        :param dst_pts: ordered lists
        :return: filtered (src_pts, dst_pts)
        """
        good_s = []
        good_t = []
        # first triangle
        if len(src_pts) > 100:
            s_range = len(src_pts) * 3  # for good quality - old front and new front
        else:
            s_range = len(src_pts) * 100  # for bad quality - new back
        for x in range(s_range):
            # must be len(src_pts) >= 3
            rand1 = random.randint(1, len(
                src_pts) - 1 - 1)  # 0+1<r1<-1  -1 - to prevent out of the array -1 leave 3 for rand2 and rand3
            rand2 = random.randint(0, rand1 - 1)  # from 0-r1
            rand3 = random.randint(rand1 + 1, len(src_pts) - 1)  # r1-l
            p1 = src_pts[rand1]
            p2 = src_pts[rand2]
            p3 = src_pts[rand3]
            if (p2[0] - p1[0]) == 0 or (p2[1] - p1[1]) == 0:  # devision by zero
                continue
            pp1 = dst_pts[rand1]
            pp2 = dst_pts[rand2]
            pp3 = dst_pts[rand3]
            d1 = distance.euclidean(p1, p2)
            d2 = distance.euclidean(p2, p3)
            d3 = distance.euclidean(p3, p1)
            dd1 = distance.euclidean(pp1, pp2)
            dd2 = distance.euclidean(pp2, pp3)
            dd3 = distance.euclidean(pp3, pp1)

            if dd1 == 0 or dd2 == 0 or dd3 == 0:  # devision by zero
                continue
            di1 = abs(d1 / dd1 - d2 / dd2)
            di2 = abs(d2 / dd2 - d3 / dd3)
            # di3 = abs(d1 / dd1 - d3 / dd3)
            # 2 of 3 is enough
            # or (di2 < 0.01 and di3 < 0.01) or (di3 < 0.01 and di1 < 0.01))\
            # check equality and not collinearity
            if di1 < height_threshold and di2 < height_threshold and (di1 < 0.005 and di2 < 0.005) and \
                    abs((p3[0] - p1[0]) / (p2[0] - p1[0]) - (p3[1] - p1[1]) / (p2[1] - p1[1])) > 0.2:

                good_s.append(src_pts[rand1])
                good_s.append(src_pts[rand2])
                good_s.append(src_pts[rand3])
                good_t.append(dst_pts[rand1])
                good_t.append(dst_pts[rand2])
                good_t.append(dst_pts[rand3])
                # remove from larges to smalles indexes
                a = sorted((rand1, rand2, rand3), reverse=True)
                # remove from src_pts 3 points but enshure that at least 3 points leave
                if len(src_pts) - 3 == 2:
                    a = a[:-1]
                if len(src_pts) - 3 == 1:
                    a = a[:-2]
                if len(src_pts) - 3 >= 1:
                    src_pts = [x for i, x in enumerate(src_pts) if i not in a]
                # break
        # print("hm", len(good_s))
        # faster way to search
        if len(good_s) >= 3:  # ok we have base
            # filter on base and new founded
            for x in range(len(src_pts) * 10):
                # get base triangle
                # print(len(good_s) - 1 - 2)
                rand_good1 = random.randint(1, len(
                    good_s) - 1 - 1)  # 0+1<r1<-1  -1 - to prevent out of the array -1 leave 1 for rand2 and rand3
                rand_good2 = random.randint(0, rand_good1 - 1)  # from 0-r1
                rand_good3 = random.randint(rand_good1 + 1, len(good_s) - 1)  # r1-l
                s1 = good_s[rand_good1]
                s2 = good_s[rand_good2]
                s3 = good_s[rand_good3]
                t1 = good_t[rand_good1]
                t2 = good_t[rand_good2]
                t3 = good_t[rand_good3]
                # get random point
                if len(src_pts) == 0:
                    break
                if len(src_pts) == 1:
                    rand1 = 0
                else:
                    rand1 = random.randint(0, len(src_pts) - 1)
                p1 = src_pts[rand1]
                pp1 = dst_pts[rand1]

                d1 = distance.euclidean(p1, s1)
                d2 = distance.euclidean(p1, s2)
                d3 = distance.euclidean(p1, s3)
                dd1 = distance.euclidean(pp1, t1)
                dd2 = distance.euclidean(pp1, t2)
                dd3 = distance.euclidean(pp1, t3)
                if dd1 == 0 or dd2 == 0 or dd3 == 0:  # devision by zero
                    continue
                di1 = abs(d1 / dd1 - d2 / dd2)
                di2 = abs(d2 / dd2 - d3 / dd3)
                # di3 = abs(d1 / dd1 - d3 / dd3)
                if di1 < 0.01 and di2 < 0.01:  # or (di2 < 0.1 and di3 < 0.1) or (di3 < 0.1 and di1 < 0.1):
                    good_s.append(src_pts[rand1])
                    good_t.append(dst_pts[rand1])
                    del src_pts[rand1]
                    del dst_pts[rand1]
                    # print("wow")

        src_pts = [[x] for x in good_s]
        dst_pts = [[x] for x in good_t]

        # draw_pts = np.array([good_s], np.int32)
        # cv.drawContours(image, draw_pts, 0, (0, 255, 0))

        return src_pts, dst_pts

    def match(self, image, add_m_ratio: float = 0, filt: bool = True) -> (int, list or None, list or None):
        """
         Match and filter matches

        :param image:
        :param add_m_ratio:
        :param filt:
        :return: points count, src_pts, dst_pts
        """
        # prepare random
        random.seed(1)

        if image is None or len(self.desc2) == 0:
            logger.error('Kaze.match: Could not open or find the image!')
            return 0, None, None

        kpts1, desc1 = self.akaze.detectAndCompute(image, None)
        nn_matches = self.matcher.knnMatch(desc1, self.desc2, 2)

        matched1 = []
        matched2 = []

        # bug of knnMatch
        if len(nn_matches) < 2 or len(nn_matches[0]) != 2 or len(nn_matches[1]) != 2:
            return 0, None, None

        for m, n in nn_matches:
            if m.distance < (self.nn_match_ratio + add_m_ratio) * n.distance:
                matched1.append(kpts1[m.queryIdx])
                matched2.append(self.kpts2[m.trainIdx])

        # ordered lists
        src_pts = [i.pt for i in matched1]
        dst_pts = [i.pt for i in matched2]

        # print("before", len(src_pts))

        if filt and 3 <= len(src_pts) <= 100:
            src_pts, dst_pts = self._filter(src_pts, dst_pts, self.orig_shape[0])

        src_pts = np.array(src_pts, np.float32)
        dst_pts = np.array(dst_pts, np.float32)

        # print("after", len(src_pts))

        return len(src_pts), src_pts, dst_pts

    # private
    def _affine_transformation(self, img, src_pts, dst_pts, scale):
        pts1 = np.float32(src_pts[:3])
        pts2 = np.float32(dst_pts[:3])
        m = cv.getAffineTransform(pts1, pts2)
        img = cv.warpAffine(img, m, (int(self.orig_shape[1] * scale), int(self.orig_shape[0] * scale)))
        return img, m

    # private
    def _transform(self, img, add_m_ratio: float = 0, scale: float = 1) -> (np.array or None, list or None, int):
        """
        1) match to original image keypoints
        2) find transformation matrix - affine or homography
        3) apply it to work image

        :param img:
        :param add_m_ratio:
        :param scale:
        :return: image, affine or homo matrix, count
        """
        img_save = img
        (count, src_pts, dst_pts) = self.match(img, add_m_ratio)
        # src_pts = np.float32(src_pts).reshape(-1, 1, 2)
        # dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
        if count < 3:
            return None, None, count

        # if count < 50:
        #     sys.stderr.write("Error: Not enough matches")
        #     return None, None, count
        # extract the matched keypoints
        # src_pts = np.float32([i.pt for i in matched1]).reshape(-1, 1, 2)
        # dst_pts = np.float32([i.pt for i in matched2]).reshape(-1, 1, 2)
        if count > 40:  # Homography
            # find homography matrix and do perspective transform
            m, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 2)  # 2
            img = cv.warpPerspective(img, m, (int(self.orig_shape[1] * scale), int(self.orig_shape[0] * scale)))
        else:  # Affine
            img, m = self._affine_transformation(img, src_pts, dst_pts, scale)
            # pts1 = np.float32(src_pts[:3])
            # pts2 = np.float32(dst_pts[:3])
            # M = cv.getAffineTransform(pts1, pts2)
            # img = cv.warpAffine(img, M, (int(self.img_orig.shape[1] * scale), int(self.img_orig.shape[0] * scale)))

        # Debug
        # print("wtf")
        # draw_pts = np.array([src_pts], np.int32)
        # cv.drawContours(img_save, draw_pts, 0, 255)
        # tmp = cv.resize(img_save, (900, 900))
        # cv.imshow('image', tmp)  # show image in window
        # cv.waitKey(0)  # wait for any key indefinitely
        # cv.destroyAllWindows()  # close window q
        # print("wtf2")

        return img, m, count

    # public and private
    @classmethod
    def prepare(self, image):
        if image is None:
            logger.error('Could not open or find the images!')
            return None

        # prepare incoming image
        if len(image.shape) == 2 or image.shape[2] == 1:
            img = image
        else:
            img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        img = cv.fastNlMeansDenoising(img, h=8, templateWindowSize=10)  # denoise tmp image
        return img

    # public
    def crop(self, image, double_crop: bool = False, random_seed: int = 1) -> any:
        """
         Double transformation of image to match template
         1) prepare: gray and fastDenoise
         2) calc transformation matrix
         3) apply transformation matrix to original image

        :param image:
        :param double_crop:
        :random_seed for retry
        :return: image or None
        """
        # prepare random
        random.seed(random_seed)

        if len(self.desc2) == 0:
            logger.error('Could not open or find the origin image!')
            return None

        img = self.prepare(image)
        if img is None:
            return None

        # Get transformation matrix by transforming tmp image
        if double_crop:
            img, m1, mcount = self._transform(img, scale=1.5)  # use prepared image
            if mcount < 3:
                return None
            img, m2, mcount2 = self._transform(img)  # use prepared image

            # Apply transformation to original image
            if mcount > 40:
                img = cv.warpPerspective(image, m1, (int(self.orig_shape[1] * 1.5), int(self.orig_shape[0] * 1.5)))
            else:
                img = cv.warpAffine(image, m1, (int(self.orig_shape[1] * 1.5), int(self.orig_shape[0] * 1.5)))
            # cv.imshow('image', img)  # show image in window
            # cv.waitKey(0)  # wait for any key indefinitely
            # cv.destroyAllWindows()  # close window q
            # Apply second
            try:
                if mcount2 > 40:
                    img = cv.warpPerspective(img, m2, (self.orig_shape[1], self.orig_shape[0]))
                else:
                    img = cv.warpAffine(img, m2, (self.orig_shape[1], self.orig_shape[0]))
            except:
                logger.info('It was impossible to apply second transformation in AKAZE.crop')
                img = img[:int(img.shape[0] / 1.5), :int(img.shape[1] / 1.5)]
                # cv.imshow('image', img)  # show image in window
                # cv.waitKey(0)  # wait for any key indefinitely
                # cv.destroyAllWindows()  # close window q

        else:  # single crop

            img, m1, mcount = self._transform(img)  # use prepared image
            if mcount < 3:
                return None
            # Apply transformation to original image
            if mcount > 40:
                img = cv.warpPerspective(image, m1, (self.orig_shape[1], self.orig_shape[0]))  # at original img
            else:
                img = cv.warpAffine(image, m1, (self.orig_shape[1], self.orig_shape[0]))
        # if img is not None:
        #     cv.imshow('image', img)  # show image in window
        #     cv.waitKey(0)  # wait for any key indefinitely
        #     cv.destroyAllWindows()  # close window q
        return img


if __name__ == '__main__':
    # obr = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/72-204-0.png'
    # obr = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/0/80-212-8.png'
    # obr = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/84-382-0.png'
    # obr = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/0/59-357-9.png'
    # obr = '/home/u2/Desktop/Untitled.png'
    # obr = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/85-383-0.png'
    obr1 = '/mnt/hit4/hit4user/Desktop/Desktop/Untitled9.png'
    obr2 = '/mnt/hit4/hit4user/Desktop/Desktop/Untitled_v2_001.png'
    # obr3 = '/mnt/hit4/hit4user/Desktop/Desktop/new_obr.png'
    # obr3 = '/mnt/hit4/hit4user/Desktop/Desktop/new_obr_v1.png'
    # obr3 = '/mnt/hit4/hit4user/Desktop/Desktop/new_obr_v2.png'
    # obr3 = '/home/u2/Desktop/back_new.png'  # /mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/1/38-48-10.png
    # obr3 = '/home/u2/Desktop/Untitled2.png'  # /mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/0/265-666-1.png
    # obr3 = '/home/u2/Desktop/back_new2.png'  # /mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/2/141-542-7.png

    # old
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/72-204-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/0/8-139-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/85-486-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/85-383-0.png'  # orig
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/84-382-0.png'  # crivaja
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/64-362-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/208-609-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/42-340-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/45-287-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/184-585-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/2/6-247-11.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/2/47-178-9.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/4-302-0.png'
    # obr old
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/275-676-1.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/66-467-1.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/91-223-1.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/34-276-1.png'
    # new
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/207-608-0.png'  # new
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/70-368-0.png'  # new
    # obr new
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/15-313-1.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/1/38-48-10.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/1/35-436-1.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/0/233-634-1.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/2/107-239-1.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/2/76-477-8.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport_and_vod/0/70-368-1.png'
    # strange rus
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/2/56-354-11.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/vodit_udostav/1/60-358-0.png'

    # CROPPER
    # o = '/home/u2/Pictures/bottom_pass3.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/3/'
    o = '/home/u2/Downloads/examples/2/PNG0.png'
    p = '/home/u2/Downloads/examples/2/'
    obr = cv.imread(o, cv.IMREAD_GRAYSCALE)
    cropper = KazeCropper(obr, width=900, nn_match_ratio=0.76)
    # IMAGE

    import os
    for i, filename in enumerate(os.listdir(p)):
        if 0 == i or i == 2:
            continue
        print(i, p + filename)
        image = cv.imread(p + filename, cv.IMREAD_COLOR)
        # image = cv.fastNlMeansDenoising(image, h=10, templateWindowSize=5)  # denoise edges
        # print(cropper.match(image)[0])
        img = cropper.crop(image, double_crop=False)
        if img is None:
            print("none")
            continue
        # tmp = cv.resize(obr, (400, 400))
        cv.imshow('image', img)  # show image in window
        cv.waitKey(0)  # wait for any key indefinitely
        cv.destroyAllWindows()  # close window q

    # image = cv.imread(obr1, cv.IMREAD_COLOR)
    # cropper1 = KazeCropper(image)
    # image = cv.imread(obr2, cv.IMREAD_COLOR)
    # cropper2 = KazeCropper(image)
    # image = cv.imread(obr3, cv.IMREAD_COLOR)
    # image = cv.fastNlMeansDenoising(image, h=8, templateWindowSize=10)  # denoise edges
    # cropper3 = KazeCropper(image)
    # img_orig = image

    # image = cv.imread(p, cv.IMREAD_COLOR)
    # image = cv.fastNlMeansDenoising(image, h=8, templateWindowSize=10)  # denoise edges
    # image = cropper3.crop(image)

    # print(cropper1.match(image)[0])
    # print(cropper2.match(image)[0])
    # img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # c, m1, m2 = cropper3.match(img)
    # # image, _, _ = cropper3.transform(image)
    # # # print(image)
    # # tmp = cv.resize(image, (900, 900))
    # # cv.imshow('image', tmp)  # show image in window
    # # cv.waitKey(0)  # wait for any key indefinitely
    # # cv.destroyAllWindows()  # close window q
    # # exit(0)
    # print(c)
    # # src_pts = np.float32([i.pt for i in m1])
    # # dst_pts = np.float32([i.pt for i in m2])
    # src_pts = [i.pt for i in m1]
    # dst_pts = [i.pt for i in m2]
    # # print(src_pts[0])
    # # src_pts = np.float32([i.pt for i in m1]).reshape(-1, 1, 2)
    # # # print(src_pts[0])
    # # dst_pts = np.float32([i.pt for i in m2]).reshape(-1, 1, 2)  # insert every pair [x,y] in list = [[x,y]]
    # # #
    # # src_pts = [[float(x[0][0]), float(x[0][1])] for x in src_pts]
    # # dst_pts = [[float(x[0][0]), float(x[0][1])] for x in dst_pts]
    #
    # src_pts, dst_pts = KazeCropper.filter(src_pts, dst_pts)
    #
    # # M = None
    # if len(src_pts) > 30:
    #     M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 2)  # homography transformation
    #     print(M)
    #     if M is not None:
    #         image = cv.warpPerspective(image, M, (img_orig.shape[1], img_orig.shape[0]))
    # else:
    # # if True:  # len(src_pts) == 3 or M is not None:
    #     # Affine transformation
    #     pts1 = np.float32(src_pts[:3])
    #     pts2 = np.float32(dst_pts[:3])
    #     # print(good_s[0][0])
    #     M = cv.getAffineTransform(pts1, pts2)
    #     print(M)
    #     image = cv.warpAffine(image, M, (int(img_orig.shape[1]*1.5), int(img_orig.shape[0]*1.5)))
    #     tmp = cv.resize(image, (900, 900))
    #     cv.imshow('image', tmp)  # show image in window
    #     cv.waitKey(0)  # wait for any key indefinitely
    #     cv.destroyAllWindows()  # close window q
    #     from cnn.shared_image_functions import fix_angle, get_lines_c
    #     image = fix_angle(image, get_lines_c)
    #     tmp = cv.resize(image, (900, 900))
    #     cv.imshow('image', tmp)  # show image in window
    #     cv.waitKey(0)  # wait for any key indefinitely
    #     cv.destroyAllWindows()  # close window q
    #     import pytesseract
    #     print(pytesseract.image_to_data(image))
    #     # exit(0)
    #     # ANGLE
    #     # theta1 = np.arctan(good_s[1][1] - good_s[0][1] / good_s[1][0] - good_s[0][0]) # s[1].y - s[0].y / s[1].x - s[0].x
    #     # theta2 = np.arctan(good_t[1][1] - good_t[0][1] / good_t[1][0] - good_t[0][0]) # t[1].y - t[0].y / t[1].x - t[0].x
    #     # theta = (theta2 - theta1) * 180 / np.pi
    #     # if theta < 0:
    #     #     theta = theta + 360
    #     #
    #     # if theta <= 90:
    #     #     theta_ = theta
    #     # elif theta <= 180:
    #     #     theta_ = (180 - theta)
    #     # elif theta <= 270:
    #     #     theta_ = (theta - 180)
    #     # elif theta <= 360:
    #     #     theta_ = (360 - theta)
    #     # else:
    #     #     print("Atan2 angle range violated")
    #     # print(theta1, theta2, theta_)
    #     #
    #     # # CALC SCALE!
    #     # (height, width, channels) = image.shape  # rows_y, cols_x
    #     # dy = width * np.sin(theta_) + height * np.cos(theta_)
    #     # dx = height * np.sin(theta_) + width * np.cos(theta_)
    #     #
    #     # pts2 = np.float32(good_t[1])
    #     # print(pts2[0], pts2[1])
    #     # rotation = cv.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), theta, 1)
    #     # image = cv.warpAffine(image, rotation, (int(dy), int(dx)))
    #     # # from cnn.shared_image_functions import fix_angle, get_lines_c
    #     # # image = fix_angle(image, get_lines_c)
    #     # tmp = cv.resize(image, (900, 900))
    #     # cv.imshow('image', tmp)  # show image in window
    #     # cv.waitKey(0)  # wait for any key indefinitely
    #     # cv.destroyAllWindows()  # close window q
    #
    #     # rotation = cv.GetRotationMatrix2D(image.Point{int(equals[1].b.keypoint.X), int(equals[1].b.keypoint.Y)}, theta, 1)
    #
    # # src_pts = [[float(x[0][0]), float(x[0][1])] for x in good]
    # # src_pts = np.array([good], np.int32)
    # # print(src_pts)
    # # # print(src_pts)
    #
    # # print(src_pts)
    # # for m
    #
    # #
    # # image = cropper.crop(image)
    # if image is not None:
    #     # tmp = cv.resize(image, (900, 900))
    #     cv.imshow('image', image)  # show image in window
    #     cv.waitKey(0)  # wait for any key indefinitely
    #     cv.destroyAllWindows()  # close window q

    # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # tmp = cv.resize(image, (900,900))
