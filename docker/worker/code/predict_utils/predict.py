from rq import Queue
from logger import logger
import redis
import uuid
import pickle
import time
import json
import cv2 as cv
# own
from config import redis_host, redis_port


def request_to_ner(queue, red, uuidstr: str, image_or_images, fun: str) -> tuple or None:
    """"""
    p_image: bytes = pickle.dumps(image_or_images, protocol=4)
    # j: Job =
    queue.enqueue(fun, p_image, uuidstr, description='file')

    re = None  # return
    time.sleep(1)
    timeout = time.time() + 60 * 1  # 1 minute to try
    while time.time() < timeout:
        try:

            if red.exists(uuidstr):
                r = red.get(uuidstr)
                if r is not None:
                    r = json.loads(r)
                    if r["status"] == "ready":
                        re = r["result"]

                        logger.debug("predict received" + str(r))
                    break

        except Exception:
            return re

        time.sleep(0.5)  # every 1/2 second
    return re


def predict(im_resized, gray_resized_not_cropped_angle_fixed) -> (
int or None, str or None, int or None):  # , np.array or None
    """
    Class 0 - passport, 1 - pts, 2 - photo, 3 - driving_license 4 - passport and vd
    subClass - 'passport_main' or 'passport_other' or None
    rotation - for passport

    :param im_resized:
    :param gray_resized_not_cropped_angle_fixed:
    :return: Class, subClass, rotation
    """
    logger.info("Start main predict")

    # Подключение для получения результатов обработки
    # NER 1 and NER 2: four connections - red_x=receive, queue_x=send
    with redis.Redis(host=redis_host, port=6379, db=3) as red_1, \
            redis.Redis(host=redis_host, port=6379, db=5) as red_2:

        # Подключение для создания очереди в Redis с помощью python-rq
        queue_1 = Queue('nn_doctype', connection=redis.Redis(host=redis_host, port=redis_port, db=2))
        queue_2 = Queue('nn_second', connection=redis.Redis(host=redis_host, port=redis_port, db=4))
        try:
            # id for our task
            uuidstr = str(uuid.uuid4().hex)

            # PASSPORT PTS OR PHOTO
            re = request_to_ner(queue_1, red_1, uuidstr, gray_resized_not_cropped_angle_fixed,
                                'predict_utils.proxies.doctype.predict_px')  # NN PASSPORT PTS OR PHOTO
            pass_photo_pts_re = re[0]  # in list

            if pass_photo_pts_re is None:
                raise Exception("predict - pass_photo_pts_re is None")

            pass_photo_pts_re = int(pass_photo_pts_re)

            if pass_photo_pts_re == 0 or pass_photo_pts_re == 4:  # passport

                logger.info("Send NN second request")
                # ORIENTATION AND MAIN PASSPORT PAGE
                re = request_to_ner(queue_2, red_2, uuidstr, im_resized,
                                    'predict_utils.proxies.second.pr_orientation_and_passmainpage')  # NN Orientation

                if re is None:
                    logger.info("Send NN second response == None")
                    return pass_photo_pts_re, None, None  # , not_resized_cropped  # passport

                (r_rot, re_m) = re

                if re_m == 0:
                    subclass = 'passport_main'
                elif re_m == 1:
                    subclass = 'passport_other'
                else:
                    subclass = 'passport_main'
                    logger.error("Error: from predict_utils.predict import predict: re_m is not in [0,1]")

                logger.info("Main predict end:" + str(pass_photo_pts_re) + ' ' + str(subclass))
                return pass_photo_pts_re, subclass, r_rot  # , not_resized_cropped  # passport

            else:  # driving license or passport_and_drivinglicense
                logger.info("Main predict end:" + str(pass_photo_pts_re))
                return pass_photo_pts_re, None, None
        finally:
            queue_1.connection.close()
            queue_2.connection.close()


def predict_is_text(image) -> bool:
    """
    TODO: None will be False
    :param image: original
    :return:
    """
    # TODO: send original image and preprocess at server side ?
    siz = 576
    img = cv.resize(image, (round(siz // 2 // 2), siz // 2))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    logger.info("Start predict_text_or_not")
    # Подключение для получения результатов обработки
    with redis.Redis(host=redis_host, port=redis_port, db=7) as red:
        # Подключение для создания очереди в Redis с помощью python-rq
        queue = Queue('nn_second', connection=redis.Redis(host=redis_host, port=redis_port, db=4))
        try:
            # id for our task
            uuidstr = str(uuid.uuid4().hex)
            logger.info("Text or Not sending")
            re = request_to_ner(queue, red, uuidstr, img, 'predict_utils.proxies.second.pr_text_or_not')
            if re is None:
                logger.error("Send NN second response == None")
                return False

            re = re[0]  # 1 or 0
            logger.info("Result of Text or Not prediction: {}".format(str(re)))
            return bool(re)
        finally:
            queue.connection.close()


def predict_is_handwrited(images: tuple) -> tuple:
    """
    TODO: None will be False
    :param images:  144 x 144 with 3 channels
    :return:
    """
    # siz = 576
    # img = cv.resize(image, (round(siz // 2 // 2), (round(siz // 2 // 2))
    logger.info("Start predict_handwrited_or_not")
    # Подключение для получения результатов обработки
    with redis.Redis(host=redis_host, port=redis_port, db=8) as red:
        # Подключение для создания очереди в Redis с помощью python-rq
        queue = Queue('nn_second', connection=redis.Redis(host=redis_host, port=redis_port, db=4))
        try:
            # id for our task
            uuidstr = str(uuid.uuid4().hex)
            logger.info("Handwriting or Not sending")
            re = request_to_ner(queue, red, uuidstr, images, 'predict_utils.proxies.second.pr_handwriting_or_not')
            if re is None:
                logger.error("Send NN second response == None")
                return (False,)

            logger.info("Result of Handwriting or Not prediction: {}".format(str(re)))
            return tuple((bool(x) for x in re))
        finally:
            queue.connection.close()

# if __name__ == '__main__':
# img = cv.imread('/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/2/41-172-0.png')
# p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/2/47-178-0.png'
# img = cv.imread(p)
# _, _, img2 = predict(img)
# cv.imshow('image', img2)  # show image in window
# cv.waitKey(0)  # wait for any key indefinitely
# cv.destroyAllWindows()  # close window q
