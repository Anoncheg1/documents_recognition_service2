import numpy as np
import typing
# own
from predict_utils.classificator_cnn_tf23 import Classifier, M

hw_or_not = Classifier(M.HW_OR_NOT, channes=3)


def predict(images: tuple) -> tuple:
    """

    :param images: gray resized to (round(siz // 2 // 2), siz // 2)
    :return: 1 - text, 0 - not text
    """
    ret = []
    for im in images:
        _, logits1 = hw_or_not.predict(im)

        val = np.mean(logits1, axis=0)
        # print(val)
        if val > 0.5:
            val = 1  # hw
        else:
            val = 0  # not
        ret.append(val)
    return tuple(ret)
