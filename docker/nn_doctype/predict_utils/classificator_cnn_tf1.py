# model boundeed
import os
import numpy as np
import tensorflow as tf
# own
from predict_utils.hnet_model_tf1.mnist_model import deep_mnist  # model


class M:
    PASSPORT_PAGE = 1
    PASSPORT_MAIN = 2
    ORIENTATION_PASSPORT = 3
    PASS_PHOTO_PTS = 4
    P_WITHOUT_VD = 5

# categories = {
#     M.PASSPORT_PAGE: keras.utils.to_categorical(range(len(paths_passport))),
#     M.PASSPORT_MAIN: keras.utils.to_categorical(range(2)),
#     M.ORIENTATION_PASSPORT: keras.utils.to_categorical(range(4)),
#     M.PASS_PHOTO_PTS: keras.utils.to_categorical(range(len(all_classes) - 1))  # passport_and_vod - not separate class
# }


models = {
    # M.PASS_PHOTO_PTS: 'selected_models/pass_photo_pts_model/model.ckpt'  # 0.973432518597237 # bad with 4 and 0 # 5 classes
    # M.PASS_PHOTO_PTS: 'selected_models/pass_photo_pts_model2_1/model.ckpt',  # 4 classes
    # M.P_WITHOUT_VD: 'selected_models/p_without_vd2_1/model.ckpt',  # 2 classes 0.958 0.917
    # M.PASS_PHOTO_PTS: 'selected_models/pass_photo_pts_model2_2/model.ckpt',  # 5 classes 0.962 0.917
    M.PASS_PHOTO_PTS: 'selected_models/pass_photo_pts_model/model.ckpt',  # 5 classes 0.93
}


def hinge_loss(logits, labels):
    all_ones = np.ones_like(labels[0])
    labels = 2 * np.array(labels) - all_ones  # to -1 1
    losses = all_ones - labels * logits
    return losses


class Classifier:
    """MyClass.i and MyClass.f are valid attribute references"""

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
        #return int(np.argmax(av_logits, axis=-1))

    @staticmethod
    def calc_orientation(ri0: list, ri1: list, ri2: list, ri3: list) -> int:
        """ sum predicted ORIENTATIONS from rotated image
        [0,1,2,3,0,1,2,3][3:7] = [3, 0, 1, 2] - first order
        [0,1,2,3,0,1,2,3][2:6] = [2, 3, 0, 1]- second order for rotated by clockwise
        """
        list_r = [0, 1, 2, 3, 0, 1, 2, 3]

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

    def __init__(self, predict_type: int):
        self.predict_type = predict_type
        # use CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # disable eager and v2
        tf.compat.v1.disable_v2_behavior()

        # BUILD MODEL
        class args:
            pass

        args.n_epochs = 90
        args.batch_size = 1  # 26
        args.learning_rate = 0.006
        args.std_mult = 0.7
        args.delay = 12
        args.phase_preconditioner = 7.8
        args.filter_gain = 2
        args.filter_size = 5
        args.n_rings = 4
        args.n_filters = 12
        args.is_classification = True
        args.dim = 144
        args.crop_shape = 0
        args.n_channels = 1
        if predict_type == M.PASS_PHOTO_PTS:
            args.n_classes = 5 #4
        elif predict_type == M.P_WITHOUT_VD:
            args.n_classes = 2
        args.lr_div = 13.

        graph = tf.Graph()
        with graph.as_default():
            self.x = tf.compat.v1.placeholder(tf.float32, [args.batch_size, 20736], name='x')
            self.train_phase = tf.compat.v1.placeholder(tf.bool, name='train_phase')
            # BUILD
            self.output = deep_mnist(args, self.x, train_phase=self.train_phase)
            saver = tf.compat.v1.train.Saver()

        self.sess = tf.compat.v1.Session(graph=graph)
        print('Restoring model parameters ...')
        model_path = os.path.join(os.getcwd(), models[predict_type])
        saver.restore(self.sess, model_path)

    def predict(self, img) -> (int, list):
        """

        :param img:
        :return:
            int - classificator result
            float - self made error meter
            float - hinge loss
            list - logits

        """
        im = (255 - img)  # range[0,1] - 1 black 0 white
        im = im / 255.0
        batch = im.reshape((1, -1))
        logits = self.sess.run(self.output, feed_dict={self.x: batch, self.train_phase: False})
        i = np.argmax(logits, axis=-1)[0]

        return i, list(logits[0])


if __name__ == '__main__':
    pass



    #
    # class args:
    #     pass
    #
    #
    # args.n_epochs = 100
    # args.batch_size = 1  # 26
    # args.learning_rate = 0.006
    # args.std_mult = 0.7
    # args.delay = 12
    # args.phase_preconditioner = 7.8
    # args.filter_gain = 2
    # args.filter_size = 5
    # args.n_rings = 4
    # args.n_filters = 12
    # args.is_classification = True
    # args.dim = 144
    # args.crop_shape = 0
    # args.n_channels = 1
    # args.n_classes = 5
    # args.lr_div = 13.
    #
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # # disable eager and v2
    # tf.compat.v1.disable_v2_behavior()
    #
    # graph = tf.Graph()
    # with graph.as_default():
    #     x = tf.compat.v1.placeholder(tf.float32, [args.batch_size, 20736], name='x')
    #     train_phase = tf.compat.v1.placeholder(tf.bool, name='train_phase')
    #     # BUILD
    #     output = deep_mnist(args, x, train_phase=train_phase)
    #     saver = tf.compat.v1.train.Saver()
    #
    # sess = tf.compat.v1.Session(graph=graph)
    #
    #
    # saver.restore(sess, '/home/u2/PycharmProjects/HNet-my/MNIST-rot/checkpoints_test/model.ckpt')
    #
    # # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/train/passport_and_vod/3/_0_307.png'
    # # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/train/pts/1/'
    # # d = '/mnt/hit4/hit4user/PycharmProjects/cnn'
    # # test_seq = CNNSequence_all(args.batch_size, d + '/test/')
    # # batcher = minibatcher(test_seq, args.batch_size, shuffle=True)
    #
    # # for i, (X, Y) in enumerate(batcher):
    #
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/test/pts/2/_0_161.png'
    # import cv2 as cv
    # img = cv.imread(p, cv.IMREAD_GRAYSCALE)
    # im = (255 - img)  # range[0,1] - 1 black 0 white
    # im = im / 255.0
    # X = im.reshape((1, -1))
    #
    # logits = sess.run(output, feed_dict={x: X, train_phase: False})
    # print(np.argmax(logits, axis=1)[0])
    # exit(0)




        # # c = Classifier(M.PASS_PHOTO_PTS)
        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # # disable eager and v2
        # tf.compat.v1.disable_v2_behavior()
        #
        #
        # class args:
        #     pass
        #
        #
        #
        # args.n_epochs = 200
        # args.batch_size = 1  # changed
        # args.learning_rate = 0.0004
        # args.std_mult = 0.7
        # args.delay = 12
        # args.phase_preconditioner = 7.8
        # args.filter_gain = 2
        # args.filter_size = 5
        # args.n_rings = 4
        # args.n_filters = 8
        # args.is_classification = True
        # args.dim = 144
        # args.crop_shape = 0
        # args.n_channels = 1
        # args.n_classes = 4
        # args.lr_div = 10.
        #
        # graph = tf.Graph()
        # with graph.as_default():
        #     x = tf.compat.v1.placeholder(tf.float32, [args.batch_size, 20736], name='x')
        #     train_phase = tf.compat.v1.placeholder(tf.bool, name='train_phase')
        #     # BUILD
        #     output = deep_mnist(args, x, train_phase=train_phase)
        #     saver = tf.compat.v1.train.Saver()
        #
        # sess = tf.compat.v1.Session(graph=graph)
        # print('Restoring model parameters ...')
        # saver.restore(sess, models[M.PASS_PHOTO_PTS])












    # import cv2 as cv
    # #
    # #
    # from predict_utils.cvutils import prepare
    # # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # # img = cv.imread('/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/2/41-172-0.png')
    # # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/2/47-178-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/test/photo/2/_0_44.png'
    # # p = '/home/u2/Desktop/5.jpg'
    # img = cv.imread(p, cv.IMREAD_GRAYSCALE)
    # # img = cv.imread(p)
    # # img = cv.resize(img, (144, 144))
    # # print("wtf")
    # # (img_passport, img_dockt, _) = prepare(img, rate=1)
    # # #     model = keras.models.load_model('/mnt/hit4/hit4user/PycharmProjects/rec2/selected_models/cnn_trained_model2019-09-12_123611.826453.h5')
    # # #     im = img_dockt
    # #
    # #     # tmp = cv.resize(im, (900, 900))
    # #     cv.imshow('image', img_dockt)  # show image in window
    # #     cv.waitKey(0)  # wait for any key indefinitely
    # #     cv.destroyAllWindows()  # close window q
    # #     #
    # #     #
    # #     # print(c.predict(im))
    # # print(img_dockt.shape)
    # im = (255 - img)
    # # im = (255 - img_dockt)  # range[0,1] - 1 black 0 white
    # im = im / 255.0
    # # if self.predict_type != M.PASSPORT_PAGE:
    # # im = im.reshape(im.shape + (1,))
    # # z = np.zeros_like(im)
    #
    # batch = np.array([im])
    # batch = batch.reshape((1, -1))
    # # print(batch)
    # logits = sess.run(output, feed_dict={x: batch, train_phase: True})
    # i = np.argmax(logits, axis=-1)[0]
    # print(i, logits)
    # #
    # # c = Classifier(M.PASS_PHOTO_PTS)
    # # print(c.predict(img))
    # exit(0)

#     p = '/home/u2/Desktop/2.jpg'
#     img = cv.imread(p)
#     img = cv.resize(img, (144, 144))
#     (img_passport, img_dockt, _) = prepare(img, rate=1)
#     print(c.predict(img_dockt))
# #

    # im = (255 - img_dockt)  # range[0,1] - 1 black 0 white
    # im = im / 255.0
    # # if self.predict_type != M.PASSPORT_PAGE:
    # # im = im.reshape(im.shape + (1,))
    # # z = np.zeros_like(im)
    #
    # batch = np.array([im])
    # batch = batch.reshape((1, -1))
    # print(batch)
    # logits = sess.run(output, feed_dict={x: batch, train_phase: True})
    # print(logits)
    # logits = 1 / (1 + np.exp(-logits))  # sigmoid
    #
    # # a = aa().predict(img_dockt, M.PASS_PHOTO_PTS)
    # print(logits)



    # p = '/home/u2/Desktop/1.jpg'
    # img = cv.imread(p)
    # img = cv.resize(img, (144, 144))
    # (img_passport, img_dockt, _) = prepare(img, rate=1)
    # # #     model = keras.models.load_model('/mnt/hit4/hit4user/PycharmProjects/rec2/selected_models/cnn_trained_model2019-09-12_123611.826453.h5')
    # # #     im = img_dockt
    # #
    # #     # tmp = cv.resize(im, (900, 900))
    # cv.imshow('image', img_dockt)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window q
    # #     #
    # #     #
    # #     # print(c.predict(im))
    # #
    # #     c = Classifier(M.PASS_PHOTO_PTS)
    # #
    #
    # im = (255 - img_dockt)  # range[0,1] - 1 black 0 white
    # im = im / 255.0
    # # if self.predict_type != M.PASSPORT_PAGE:
    # # im = im.reshape(im.shape + (1,))
    # # z = np.zeros_like(im)
    #
    # batch = np.array([im])
    # batch = batch.reshape((1, -1))
    # print(batch)
    # logits = sess.run(output, feed_dict={x: batch, train_phase: True})
    # print(logits)
    # logits = 1 / (1 + np.exp(-logits))  # sigmoid
    #
    # # a = aa().predict(img_dockt, M.PASS_PHOTO_PTS)
    # print(logits)





    # im = (255 - im)  # range[0,1]
    # im = im / 255.0
    # im = im.reshape(im.shape + (1,))  # channels
    # batch = np.array([im])

#     # if self.predict_type != M.PASSPORT_PAGE:
#     im = im.reshape(im.shape + (1,))
#     batch = np.array([im])
#
#     logits = model.predict(batch, batch_size=None, verbose=0, steps=None)
#     print(logits)
#     # cv.imshow('image', img2)  # show image in window
#     # cv.waitKey(0)  # wait for any key indefinitely
#     # cv.destroyAllWindows()  # close window q
