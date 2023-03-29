import logging

# from rq.worker import logger
UUID: str = None  # noqa

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger()

# Add UUID to message
old_factory = logging.getLogRecordFactory()


def record_factory(*args, **kwargs):
    global UUID
    record = old_factory(*args, **kwargs)
    if UUID is not None:
        record.msg = UUID + ":" + record.msg
    return record


logging.setLogRecordFactory(record_factory)


def set_uuid(uuid: str):
    global UUID
    UUID = uuid
