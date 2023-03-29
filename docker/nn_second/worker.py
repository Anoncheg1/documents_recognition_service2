#!/usr/bin/env python
# for custom worker
import sys
from rq import Connection, Worker, Queue, push_connection, pop_connection
from redis import Redis
import threading
from threading import Thread
import os
import time
from rq.worker import logger as lg

from debug_or_not import redis_host

# Preload libraries
import predict_utils.predict_orientation_passmainpage
import predict_utils.predict_text_or_not
import predict_utils.predict_handwriting_or_not

TIME_IN_QUEUE = 700
WAIT_FOR_THREAD = 30


class WorkerMy(Worker):
    def perform_job(self, job, queue, heartbeat_ttl=None):
        rv = job.perform()
        job._result = rv
        job.delete(pipeline=self.connection, remove_from_queue=False)

    def fork_work_horse(self, job, queue):
        os.environ['RQ_WORKER_ID'] = self.name
        os.environ['RQ_JOB_ID'] = job.id

        thread: Thread = threading.Thread(target=self.perform_job, args=(job, queue), daemon=True)
        thread.start()
        self._horse_pid = thread
        self.procline('Theaded thread {0} at {1}'.format(thread.ident, time.time()))

    def monitor_work_horse(self, job):
        self._horse_pid.join(WAIT_FOR_THREAD)  # 500 seconds
        if self._horse_pid.is_alive():
            logger.critical("monitor_work_horse: process still working after join 500")
            raise Exception


if __name__ == '__main__':
    # trying to set level
    LOG_LEVEL = 'WARNING'
    lg.setLevel(LOG_LEVEL)
    # accepted request connection
    conn = Redis(host=redis_host, port=6379, db=4, retry_on_timeout=True)  #, password=url.password

    q = Queue('nn_second', connection=conn, default_timeout=300)  # queue name - used in predict_utils/predict.py

    w = WorkerMy(q, connection=conn)
    w.work(logging_level=LOG_LEVEL)
