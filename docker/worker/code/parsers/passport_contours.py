import cv2 as cv
import numpy as np


def contours(img, i: int):
    """
    :param img: exist side effect
    :param i:
    :return: cnts, thresh
    """
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = img

    gray = cv.fastNlMeansDenoising(gray, h=(20 * (i*1)), templateWindowSize=20)  # h=40, templateWindowSize=20)

    # initialize a rectangular and square structuring kernel
    rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (13, 5))
    sqKernel = cv.getStructuringElement(cv.MORPH_RECT, (31, 31))
    #
    thresh = cv.GaussianBlur(gray, (3, 3), 0)
    thresh = cv.morphologyEx(thresh, cv.MORPH_BLACKHAT, rectKernel)
    # img2 = cv.resize(thresh, (900, 900))
    # cv.imshow("ROI", img2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # compute the Scharr gradient of the blackhat image and scale the
    # result into the range [0, 255]
    thresh = cv.Sobel(thresh, ddepth=cv.CV_32F, dx=1, dy=0, ksize=-1)
    thresh = np.absolute(thresh)
    (minVal, maxVal) = (np.min(thresh), np.max(thresh))
    thresh = (255 * ((thresh - minVal) / (maxVal - minVal))).astype("uint8")
    # apply a closing operation using the rectangular kernel to close
    # gaps in between letters -- then apply Otsu's thresholding method
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, rectKernel)
    thresh = cv.threshold(thresh, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    #

    # perform another closing operation, this time using the square
    # kernel to close gaps between lines of the MRZ, then perform a
    # series of erosions to break apart connected components
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, sqKernel)

    thresh = cv.erode(thresh, None, iterations=4)


    # during thresholding, it's possible that border pixels were
    # included in the thresholding, so let's set 5% of the left and
    # right borders to zero
    p = int(img.shape[1] * 0.03)
    thresh[:, 0:p] = 0
    thresh[:, img.shape[1] - p:] = 0




    # Remove vertical lines
    # V = cv.Sobel(thresh, cv.CV_8U, dx=1, dy=0)  # vertical lines
    # H = cv.Sobel(thresh, cv.CV_8U, dx=0, dy=1)  # horizontal lines
    # mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    # contours = cv.findContours(V, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[1]
    # height = gray.shape[0]
    # for cnt in contours:
    #     (x, y, w, h) = cv.boundingRect(cnt)
    #     # manipulate these values to change accuracy
    #     if h > height / 3 and w < 40:
    #         cv.drawContours(mask, [cnt], -1, 255, -1)
    # img2 = cv.resize(mask, (900, 900))
    # cv.imshow("ROI", img2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # mask = cv.morphologyEx(mask, cv.MORPH_DILATE, None, iterations=3)
    # img2 = cv.resize(mask, (900, 900))
    # cv.imshow("ROI", img2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()




    # find contours in the thresholded image and sort them by their
    # size
    cnts, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(mask, [contours[i]], -1, 0, -1)

    # print(len(cnts))
    # cv.drawContours(gray, cnts, -1, (255, 255, 0), 3)  # DEBUG!!

    # img2 = cv.resize(thresh, (900, 900))
    # cv.imshow("ROI", img2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # cnts = cnts
    # cnts = sorted(cnts, key=cv.contourArea, reverse=True)

    # cv.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # linek = np.zeros((27, 27), dtype=np.uint8)
    # linek[..., 13] = 1
    # x = cv.morphologyEx(thresh, cv.MORPH_OPEN, linek, iterations=1)
    # thresh_save -= x
    # img2 = cv.resize(gray, (900, 900))
    # cv.imshow('image', img2)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q

    return cnts, thresh


def contours2(gray, i: int):
    """
    :param img: exist side effect
    :param i: 0,1,2,3,4
    :return: cnts, thresh
    """
    # print("a")
    # gray = cv.fastNlMeansDenoising(gray, h=20*i, templateWindowSize=20)  # h=40, templateWindowSize=20)
    # print("b")
    # initialize a rectangular and square structuring kernel
    rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (13, 5))
    sqKernel = cv.getStructuringElement(cv.MORPH_RECT, (31, 31))
    #
    thresh = cv.GaussianBlur(gray, (1+i*8, 1+i*8), 0)
    # thresh = cv.blur(gray, (1 + i*3, 3 + i*3))
    # thresh = cv.blur(gray, (3 + i * 3, 3 + i * 3))
    # thresh = cv.bilateralFilter(gray, (15 - i * 0),80,80)
    img2 = cv.resize(gray, (900, 900))
    cv.imshow("ROI", img2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    thresh = cv.morphologyEx(thresh, cv.MORPH_BLACKHAT, rectKernel)

    # compute the Scharr gradient of the blackhat image and scale the
    # result into the range [0, 255]
    thresh = cv.Sobel(thresh, ddepth=cv.CV_32F, dx=1, dy=0, ksize=-1)
    thresh = np.absolute(thresh)
    (minVal, maxVal) = (np.min(thresh), np.max(thresh))
    thresh = (255 * ((thresh - minVal) / (maxVal - minVal))).astype("uint8")
    # apply a closing operation using the rectangular kernel to close
    # gaps in between letters -- then apply Otsu's thresholding method
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, rectKernel)
    thresh = cv.threshold(thresh, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    #
    # img2 = cv.resize(thresh, (900, 900))
    # cv.imshow("ROI", img2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # perform another closing operation, this time using the square
    # kernel to close gaps between lines of the MRZ, then perform a
    # series of erosions to break apart connected components
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, sqKernel)

    thresh = cv.erode(thresh, None, iterations=4)

    # during thresholding, it's possible that border pixels were
    # included in the thresholding, so let's set 5% of the left and
    # right borders to zero
    p = int(img.shape[1] * 0.03)
    thresh[:, 0:p] = 0
    thresh[:, img.shape[1] - p:] = 0




    # Remove vertical lines
    # V = cv.Sobel(thresh, cv.CV_8U, dx=1, dy=0)  # vertical lines
    # H = cv.Sobel(thresh, cv.CV_8U, dx=0, dy=1)  # horizontal lines
    # mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    # contours = cv.findContours(V, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[1]
    # height = gray.shape[0]
    # for cnt in contours:
    #     (x, y, w, h) = cv.boundingRect(cnt)
    #     # manipulate these values to change accuracy
    #     if h > height / 3 and w < 40:
    #         cv.drawContours(mask, [cnt], -1, 255, -1)
    # img2 = cv.resize(mask, (900, 900))
    # cv.imshow("ROI", img2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # mask = cv.morphologyEx(mask, cv.MORPH_DILATE, None, iterations=3)
    # img2 = cv.resize(mask, (900, 900))
    # cv.imshow("ROI", img2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # find contours in the thresholded image and sort them by their
    # size
    cnts, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)

    # cv.drawContours(gray, cnts, -1, (255, 255, 0), 3)  # DEBUG!!
    #
    # img2 = cv.resize(thresh, (900, 900))
    # cv.imshow("ROI", img2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return cnts, thresh


if __name__ == '__main__':
    p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/0/6-137-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/0/11-412-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/0/12-253-0.png'
    p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/0/5-13-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/prep/passport/passport_main/25-716-0.png_2.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/prep/passport/passport_main/31-273-0.png_0.png'

    img = cv.imread(p, cv.IMREAD_COLOR)
    print(img.shape)

    from cnn.shared_image_functions import crop

    img, _ = crop(img, rotate=True, rate=1)
    # for _ in range(4 - 4):
    #     timg = cv.transpose(img)
    #     img = cv.flip(timg, flipCode=1)


    # display
    tmp = cv.resize(img, (900, 900))
    cv.imshow('image', tmp)  # show image in window
    cv.waitKey(0)  # wait for any key indefinitely
    cv.destroyAllWindows()  # close window q

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # gray = cv.fastNlMeansDenoising(gray, h=20, templateWindowSize=10)  # h=40, templateWindowSize=20)
    exit(0)
    for i in range(5):

        cnts, thresh = contours2(gray.copy(), i)  # gradient by X
        print(len(cnts))
        if cnts and 50 < len(cnts) < 95:
            print("ok")