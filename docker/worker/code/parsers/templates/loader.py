import json
from logger import logger
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


def load(what_to_load: str):
    try:
        with open(curr_dir + '/' + what_to_load + '.json', 'r') as f:
            template = json.load(f)
    except ValueError:  # includes simplejson.decoder.JSONDecodeError
        logger.exception('Decoding JSON template has failed')
    except:  # includes simplejson.decoder.JSONDecodeError
        logger.exception('Fail to load template')
    return template


