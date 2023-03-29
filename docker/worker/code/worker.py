#!/usr/bin/env python
import sys
from rq import Connection, Worker, Queue, push_connection, pop_connection
from rq.registry import FailedJobRegistry, StartedJobRegistry, clean_registries
from rq.worker import logger as lg, StopRequested
from rq.job import JobStatus, Job
from redis import Redis
# import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process
import os
import time
import datetime
from rq.worker import logger as lg
from rq import worker_registration
# own
from config import redis_host, redis_port
from config import redis_get_request_socket_timeout

# Preload libraries
# own
import MainOpenCV
# seconds
TIME_IN_QUEUE = 700  # skip job if it was in queue too long
WAIT_FOR_THREAD = 20*60  # halt if thread execution time more 20m
JOB_EXPIRE = 60*60*6  # expire processed job
JOB_FAILED_SAVED = 60*60*24

executor = ThreadPoolExecutor(max_workers=1)


class WorkerMy(Worker):

    # self.connection - one connection for all queues
    def perform_job(self, job: Job, queue, heartbeat_ttl=None):
        job.ttl = JOB_EXPIRE   # how long a job will be persisted
        job.result_ttl = JOB_EXPIRE  # how long a job result will be persisted
        job.failure_ttl = JOB_EXPIRE  # how long a job result will be persisted
        # to current
        self.q_current.enqueue_job(job)
        with self.connection.pipeline() as pipeline:
            self.set_current_job_id(job.id, pipeline=pipeline)
            pipeline.execute()

        # PEROFORM
        job.started_at = datetime.datetime.utcnow()
        try:
            self.log.debug("WORKER perform!")
            rv = job.perform()
            job._result = rv

        except:  # noqa
            # job.ttl = JOB_FAILED_SAVED
            self.log.exception("Job failed:" + job.id)
            self.handle_job_failure(job, queue)
            # remove from current
            self.q_current.remove(job)

        job.ended_at = datetime.datetime.utcnow()
        self.handle_job_success(job=job,
                                queue=queue,
                                started_job_registry=queue.started_job_registry)
        # # remove from current
        self.q_current.remove(job)
        # with self.connection.pipeline() as pipeline:
        #     self.set_current_job_id(None, pipeline=pipeline)
        #     pipeline.execute()
        # to processed and set expire
        self.q_processed.enqueue_job(job, at_front=True)

    def fork_work_horse(self, job, queue):
        """called from execute_job
         Thread instead of os.fork"""

        if int(time.time()) - job.meta > TIME_IN_QUEUE:
            self.log.error("skip job with time_in_queue > 700s")  # debug
            with self.connection.pipeline() as pipeline:
                job.delete(pipeline=pipeline, remove_from_queue=False)
                pipeline.execute()
                return

        os.environ['RQ_WORKER_ID'] = self.name
        os.environ['RQ_JOB_ID'] = job.id
        # Process
        proc: Process = Process(target=self.perform_job, args=(job, queue), daemon=False)
        self.log.debug("WORKER fork Process!")
        proc.start()  # and continue
        # ThreadPool
        # task = executor.submit(self.perform_job, job, queue)
        # Thread
        # thread: Thread = threading.Thread(target=self.perform_job, args=(job, queue), daemon=True)
        # thread.start()
        self._horse_pid = proc
        # self.procline('Theaded thread {0} at {1}'.format(self._horse_pid, time.time()))

    def monitor_work_horse(self, job):
        """called from execute_job
        right after fork_work_horse"""

        try:
            # self._horse_pid.result(WAIT_FOR_THREAD)  # seconds
            self.log.debug("WORKER join for " + str(WAIT_FOR_THREAD))
            self._horse_pid.join(WAIT_FOR_THREAD)  # seconds
            self.log.debug("WORKER join expire")
            if self._horse_pid.is_alive():
                self.log.debug("WORKER steel alive after join expire")
                # job.ttl = JOB_FAILED_SAVED
                # remove from current
                self.handle_job_failure(job, queue)
                self.q_current.remove(job)
                self.log.error("monitor_work_horse: process still working after join " +
                                  str(WAIT_FOR_THREAD) + "s .Job:" + str(job.id))
                self._stop_requested = True
                self.register_death()
                self._horse_pid.terminate()
        except TimeoutError:
            # job.ttl = JOB_FAILED_SAVED
            # remove from current
            self.handle_job_failure(job, queue)
            self.q_current.remove(job)
            self.log.critical(
                "monitor_work_horse: process still working after join " + str(WAIT_FOR_THREAD) + "s .Job:" + str(
                    job.id))
            self._stop_requested = True
            self.register_death()
            self._horse_pid.terminate()


if __name__ == '__main__':
    # Switch RQ serializer to protocol 4 for python 3.7 compatibility
    from functools import partial
    from rq.serializers import DefaultSerializer
    import pickle

    # configure pickle
    DefaultSerializer.dumps = partial(pickle.dumps, protocol=4)

    LOG_LEVEL = 'DEBUG'
    lg.setLevel(LOG_LEVEL)
    while True:
        lg.info("Create Worker in __main__")
        # accepted request connection
        conn = Redis(host=redis_host, port=redis_port, db=0)  # socket_timeout=redis_get_request_socket_timeout

        q1 = Queue('in_medium', connection=conn)
        q2 = Queue('in_high', connection=conn)

        # default_result_ttl - ttl in queue.finished_job_registry
        # default_worker_ttl - expirration time of the worker
        # job_monitoring_interval - heartbeat_ttl (must be > default job timeout)
        # timeout for blocking conn get job
        w = WorkerMy([q2, q1], connection=conn, default_worker_ttl=WAIT_FOR_THREAD+10)
        w.q_current = Queue('current', connection=conn)
        w.q_processed = Queue('processed', connection=conn)

        w.work(logging_level=LOG_LEVEL, with_scheduler=False)  # heartbeat here
