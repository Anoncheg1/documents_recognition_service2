import pickle
import redis
import json
from typing import Callable
# own
from logger import logger
from debug_or_not import redis_host

redis_expire = 500  # seconds


def common_proxy(image_pickle, id_processing, predict: Callable, db_response: int):

    redis_connect = redis.Redis(host=redis_host, port=6379, db=db_response)
    logger.info("Start prediction")

    try:
        id_processing = str(id_processing)
        # Response connection to first Worker

        image = pickle.loads(image_pickle)
        res = predict(image)

        finalout = {
            "status": "ready",
            "result": [int(x) for x in res]
        }
        logger.info("res:" + str(finalout))
        redis_connect.set(id_processing, json.dumps(finalout), ex=redis_expire)
    except Exception as e:
        logger.exception("Uncatched exception in predict")
        if hasattr(e, 'message'):
            o = {"status": "exception", "description": e.message}
        else:
            o = {"status": "exception", "description": str(type(e).__name__) + " : " + str(e.args)}
        redis_connect.set(id_processing, json.dumps(o), ex=redis_expire)
        raise  # return and wait for next request

    logger.info("End prediction:" + str(res))
