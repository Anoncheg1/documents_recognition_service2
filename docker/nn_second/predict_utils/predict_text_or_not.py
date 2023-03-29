import numpy as np
# own
from predict_utils.classificator_cnn_tf23 import Classifier, M

text_or_not = Classifier(M.TEXT_OR_NOT)


def predict(image) -> tuple:
    """

    :param image: gray resized to (round(siz // 2 // 2), siz // 2)
    :return: 1 - text, 0 - not text
    """
    image_flipped_left = np.flip(image, axis=1)
    # image_flipped_down = np.flipud(image)
    image_rot180 = np.rot90(np.rot90(image))

    _, logits1 = text_or_not.predict(image)
    _, logits2 = text_or_not.predict(image_flipped_left)
    # _, logits3 = text_or_not.predict(image_flipped_down)
    _, logits4 = text_or_not.predict(image_rot180)
    # print(logits1, logits2, logits4)
    val = np.mean((logits1[0], logits2[0], logits4[0]), axis=0)
    if val > 0.3:
        val = 1
    else:
        val = 0
    # val = Classifier.calc_max_logit(
    #     (logits1[0], logits2[0], logits3[0], logits4[0]))  # [0] - in batch, [0] - double array
    # print(logits1[0][0], logits2[0][0], logits3[0][0], logits4[0][0])
    return (val,)
