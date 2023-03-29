from predict_utils.proxy_util import common_proxy
from predict_utils.predict_doctype import predict


def predict_px(image_pickle, id_processing):
    common_proxy(image_pickle, id_processing, predict, db_response=3)
