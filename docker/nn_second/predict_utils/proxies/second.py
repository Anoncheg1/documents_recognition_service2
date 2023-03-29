from predict_utils.proxy_util import common_proxy
from predict_utils import predict_orientation_passmainpage
from predict_utils import predict_text_or_not
from predict_utils import predict_handwriting_or_not


def pr_orientation_and_passmainpage(image_pickle, id_processing):
    common_proxy(image_pickle, id_processing, predict_orientation_passmainpage.predict,
                 db_response=5)


def pr_text_or_not(image_pickle, id_processing):
    common_proxy(image_pickle, id_processing, predict_text_or_not.predict,
                 db_response=7)


def pr_handwriting_or_not(pickle_images, id_processing):
    common_proxy(pickle_images, id_processing, predict_handwriting_or_not.predict,
                 db_response=8)
