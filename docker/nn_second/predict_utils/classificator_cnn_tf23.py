# SHARED BETWEEN CNN WORKERS
import os
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Model, model_from_json
# own


# SOLV Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)


from cnn.classes import paths_passport, all_classes
# from cnn.shared_image_functions import most_common
# from gcnn.convolutional import GConv2D
# from gcnn.normalization import BatchNormalization


class M:
    PASSPORT_PAGE = 1
    PASSPORT_MAIN = 2
    ORIENTATION_PASSPORT = 3
    PASS_PHOTO_PTS = 4
    TEXT_OR_NOT = 5
    HW_OR_NOT = 6


categories = {
    M.PASSPORT_PAGE: keras.utils.to_categorical(range(len(paths_passport))),
    M.PASSPORT_MAIN: keras.utils.to_categorical(range(2)),
    M.ORIENTATION_PASSPORT: keras.utils.to_categorical(range(4)),
    M.PASS_PHOTO_PTS: keras.utils.to_categorical(range(len(all_classes) - 1)),  # passport_and_vod - not separate class
    M.TEXT_OR_NOT: keras.utils.to_categorical(range(2)),
    M.HW_OR_NOT: keras.utils.to_categorical(range(2)),
}

models = {
    M.PASSPORT_PAGE: 'cnn_trained_model2019-09-11_150548.715582.h5',  #  [0-1] val_loss: 0.1797 - val_accuracy: 0.8477
    M.PASSPORT_MAIN: 'cnn_trained_model2019-09-11_155948.476527',  # [0-1] val_loss: 0.0149 - val_accuracy: 1.0000
    M.ORIENTATION_PASSPORT: 'cnn_trained_model2019-09-12_123611.826453',  # 4 range[0,1] # val_loss: 0.2999 - val_accuracy: 0.8487
    # M.TEXT_OR_NOT: 'cnn_trained_model2020-09-10 09:26:34.553480',  # 2 [0-1] binary classif val_loss: 0.1821 - val_accuracy: 0.9625
    M.TEXT_OR_NOT: 'cnn_trained_model2020-10-07 10:06:36.703748',  # 1 [0-1] binary classif loss: 0.0185 - accuracy: 0.9940 - val_loss: 0.0166 - val_accuracy: 0.9946
    # M.HW_OR_NOT: 'cnn_hw_or_not2020-12-16 10:35:50.488656' # val_loss: 0.1679 - val_accuracy: 0.9383
    M.HW_OR_NOT: 'cnn_hw_or_not2020-12-23 05:08:14.400416'  # val_loss: 0.0618 - val_accuracy: 0.9796
}


def hinge_loss(logits, labels):
    all_ones = np.ones_like(labels[0])
    labels = 2 * np.array(labels) - all_ones  # to -1 1
    losses = all_ones - labels * logits
    return losses


class Classifier:
    """MyClass.i and MyClass.f are valid attribute references"""

    save_dir = os.path.join(os.getcwd(), 'selected_models')  # must in current directory

    @staticmethod
    def rotate(img):
        img_rot1 = np.rot90(img)  # not clockwise
        img_rot2 = np.rot90(img_rot1)  # not clockwise
        img_rot3 = np.rot90(img_rot2)  # not clockwise
        return img_rot3, img_rot2, img_rot1

    # @staticmethod
    # def aver_logits(r0, r1, r2, r3) -> int:
    #     av_logits = [np.average((x[0], x[1], x[2], x[3])) for x in zip(r0, r1, r2, r3)]
    #     return av_logits
    #     return int(np.argmax(av_logits, axis=-1))
    @staticmethod
    def calc_max_logit(logits: iter) -> int:
        mean_logits = np.mean(logits, axis=0)
        return int(np.argmax(mean_logits, axis=-1))

    @staticmethod
    def calc_orientation(ri0, ri1, ri2, ri3) -> int:
        """ sum predicted orientations from rotated image"""
        list_r = [0, 1, 2, 3, 0, 1, 2, 3]
        # [0,1,2,3,0,1,2,3][3:7] = [3, 0, 1, 2] - first order
        # [0,1,2,3,0,1,2,3][2:6] = [2, 3, 0, 1]- second order for rotated by clockwise
        ri1_new = np.zeros(4)  # np.zeros_like(ri0)
        for new, old in enumerate(list_r[3:7]):
            ri1_new[old] = ri1[new]

        ri2_new = np.zeros(4)
        for new, old in enumerate(list_r[2:6]):
            ri2_new[old] = ri2[new]

        ri3_new = np.zeros(4)
        for new, old in enumerate(list_r[1:5]):
            ri3_new[old] = ri3[new]

        av_logits = np.average((ri0, ri1_new, ri2_new, ri3_new), axis=0)

        return int(np.argmax(av_logits, axis=-1))

    def __init__(self, predict_type: int, channes: int = 1):
        self.predict_type = predict_type
        self.channels = channes
        # use CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # parent_path = os.path.join(os.getcwd(), os.pardir)
        model_name = models[predict_type]
        model_path = os.path.join(Classifier.save_dir, model_name)
        # print("model folder", model_path)

        # old (full load):
        # self.model: Model = keras.models.load_model(model_path)  # , custom_objects={'GConv2D': GConv2D})
        with open(os.path.join(model_path, "model_to_json.json"), "r") as json_file:
            json_model = json_file.read()

        self.model: Model = model_from_json(json_model)  # load weights into new model
        self.model.load_weights(os.path.join(model_path, "weights.tf")).expect_partial()

        # import tensorflow_core.python.keras.backend as K
        # from tensorflow_core.python.framework.ops import get_default_graph
        # # self.model._make_predict_function()
        # self.session = K.get_session()
        # self.graph = get_default_graph()

        # self.graph.finalize()  # finalize

    def predict(self, img) -> (int, list):
        """

        :param img:
        :return:
            int - classificator result
            float - self made error meter
            float - hinge loss
            list - logits

        """
        im = img
        # im = self.prepare(img)
        # if self.predict_type != M.PASSPORT_PAGE:
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # im = cv.resize(img, (siz, siz))
        # img2 = cv.resize(im, (900, 900))
        # cv.imshow('image', img2)  # show image in window
        # cv.waitKey(0)  # wait for any key indefinitely
        # cv.destroyAllWindows()  # close window
        # im = im / 128.0  # range[0,2]
        # im = (2 - im)
        im = (255 - im)  # range[0,1] - 1 black 0 white
        im = im / 255.0
        # if self.predict_type != M.PASSPORT_PAGE:
        if self.channels == 1:
            im = im.reshape(im.shape + (1,))

        batch = np.array([im])
        # with self.session.as_default():
        logits = self.model.predict(batch, batch_size=None, verbose=0, steps=None,)
        i = np.argmax(logits, axis=-1)[0]
        # hl = hinge_loss(logits, y_passport[i])
        logits = logits.tolist()[0]
        # t = logits[i]
        # err = 0
        # if t > 0:
        #     l = [(1-abs(t - x))**2 for ii, x in enumerate(logits) if x > 0 and ii != i]
        #     if l:
        #         # print(l)
        #         err = sum(l)/len(l)
        # else:
        #     err = 0.2
        # # if i != 1:
        # #     err += 3
        # # print(err)

        # h_loss = hinge_loss(logits, np.zeros_like(categories[self.predict_type][0]))

        return i, logits


if __name__ == '__main__':
    # a= Classifier.calc_max_logit(([0, 1,1],[1,0,1]))
    # print(a)
    import cv2 as cv
    img = cv.imread('/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/2/41-172-0.png')
    print(img.shape)
    # import cv2 as cv
    # from predict_utils.cvutils import prepare
    # # prepare image
    # img = cv.imread('/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/2/41-172-0.png')
    # im_resized, gray_not_crop_not_res, not_resized_cropped = prepare(img, rate=1)
    #
    # orientation = Classifier(M.ORIENTATION_PASSPORT)
    #
    # img_rot1, img_rot2, img_rot3 = Classifier.rotate(im_resized)  # clockwise
    # #
    # re_r0 = orientation.predict(img)
    # re_r1 = orientation.predict(img_rot1)
    # re_r2 = orientation.predict(img_rot2)
    # re_r3 = orientation.predict(img_rot3)
    # del img_rot1, img_rot2, img_rot3
    # r_rot = Classifier.calc_orientation(re_r0[1], re_r1[1], re_r2[1], re_r3[1])
    # print(r_rot)

#     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#     # img = cv.imread('/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/2/41-172-0.png')
#     p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/2/47-178-0.png'
#     img = cv.imread(p)
#     im_resized, gray_not_crop_not_res, not_resized_cropped = Classifier.prepare(img, rate=1)
#     model = keras.models.load_model('/mnt/hit4/hit4user/PycharmProjects/rec2/selected_models/cnn_trained_model2019-09-12_123611.826453.h5')
#     im = im_resized
#     im = (255 - im)  # range[0,1]
#     im = im / 255.0
#     # if self.predict_type != M.PASSPORT_PAGE:
#     im = im.reshape(im.shape + (1,))
#     batch = np.array([im])
#
#     logits = model.predict(batch, batch_size=None, verbose=0, steps=None)
#     print(logits)
#     # cv.imshow('image', img2)  # show image in window
#     # cv.waitKey(0)  # wait for any key indefinitely
#     # cv.destroyAllWindows()  # close window q
