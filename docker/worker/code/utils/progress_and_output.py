import redis
import json
import uuid
from threading import Lock
from datetime import datetime
import time
# own
from logger import logger
from config import redis_resp_expire
from utils.simplify_api import simplify
from utils.doctype_by_text import mark_second_files

PROFILING = False  # if test set True


class OutputToRedis:
    """ Not used in profiling
    worker.py -> threads -> OutputToRedis  -> threads -> parse (class by uuid) -> progress inside parsers
    Instance created per request."""
    instances = {}  # request_uuid: (OutputToRedis, datetime)
    instances_lock = Lock()

    def __init__(self, id_uuid: str, conn: redis.Redis = None, do_not_mark_second=False):
        """ 1)

        :param id_uuid: id for request
        :param conn: used to feed back status
        :param do_not_mark_second: for dtype in (130, 140, 100) and (x['qc'] == 3 or x['qc'] == 4) look at page before
        """
        self.time = time.time()
        self._id_uuid = id_uuid
        self._connection = conn
        self.do_not_mark_second = do_not_mark_second
        self.inst_lock = Lock()  # lock this instance when parse parallel pages
        self._pages_list = None
        self._pages_count = 0  # we dont knew yet
        if self._connection:
            self._connection.set(id_uuid, json.dumps({"status": "waiting"}), ex=redis_resp_expire)
            # save instance
        with OutputToRedis.instances_lock:
            OutputToRedis.instances[id_uuid] = self

    def _pr(self, current_page=0, page_progress=0):
        """ with self._pages_lock - required"""
        # calc full progress
        full_progress = 0
        one_page = 100 / self._pages_count
        # with self._pages_lock:
        for p in self._pages_list:
            if p is not None:
                if p['qc'] == -1:
                    full_progress += one_page / 100 * p["page_progress"]
                else:
                    full_progress += one_page

        return {"pages": self._pages_count, "current_page": current_page, "page_progress": page_progress,
                "full_progress": round(full_progress)}

    def after_split(self, pages: int) -> None:
        """ 2) """
        self._pages_count = pages
        self._pages_list: list = [None for _ in range(pages)]  # instance attribute, ACCESS ONLY WITH LOCK
        if self._connection:
            self._connection.set(self._id_uuid, json.dumps(
                {"status": "processing", "progress": self._pr()}), ex=redis_resp_expire)

    def exception(self, descr: str):
        message = {"status": "exception", "description": descr}
        if self._connection:
            self._connection.set(self._id_uuid, json.dumps(message), ex=redis_resp_expire)

    @staticmethod
    def progress(obj, progress: int, doc_type: int = None):
        """  TODO: add file uuid in progress

        :param obj: doc_types.DocTypeDetected
        :param progress: 0-100
        :param doc_type: we pass doc_type to mark that we are shure. obj.doc_type may be changed.
        :return:
        """
        if obj.request_uuid is None:
            print("progress:", progress)  # test mode
        # getting instance to self
        with OutputToRedis.instances_lock:
            if obj.request_uuid not in OutputToRedis.instances:
                return
            self: OutputToRedis = OutputToRedis.instances[obj.request_uuid]
        # setting lock and work with _pages_lost
        with self.inst_lock:
            if self._pages_list[obj.page_num] is not None:  # increase
                self._pages_list[obj.page_num]['page_progress'] = progress
            elif doc_type is not None and doc_type != 0:
                self._pages_list[obj.page_num] = {'document_type': doc_type, 'page_progress': progress, 'qc': -1}
            else:
                self._pages_list[obj.page_num] = {'page_progress': progress, 'qc': -1}

        if self.do_not_mark_second:
            final_pages: list = self._pages_list
        else:
            final_pages: list = mark_second_files(self._pages_list)

        message = {"status": "processing", "progress": self._pr(obj.page_num, progress),
                   "pages": final_pages}
        jstr = json.dumps(message, ensure_ascii=False, indent=4)
        try:
            if self._connection:
                self._connection.set(self._id_uuid, jstr, ex=redis_resp_expire)
        except redis.exceptions.ConnectionError as e:  # not important
            logger.warn("progress redis.exceptions.ConnectionError " + str(e.args))

    @staticmethod
    def get_text_doc_orientation(request_uuid: str) -> int or None:
        if request_uuid is None:  # test mode
            return None

        with OutputToRedis.instances_lock:
            if request_uuid not in OutputToRedis.instances:
                return None
            self: OutputToRedis = OutputToRedis.instances[request_uuid]

        with self.inst_lock:
            for x in self._pages_list:
                if x is not None and 'text_doc_orientation' in x and x['text_doc_orientation'] is not None:
                    return x['text_doc_orientation']
        return None

    def page_completed(self, page: int, ret_dict: dict, simplify_flag: bool = False) -> str:
        """
        1) simplify 2) generate file_uuid 3) save new result 4) mark second 5) send progress
        :param page:
        :param ret_dict:
        :param simplify_flag:
        :return: file uuid
        """

        if simplify_flag:
            try:
                ret_dict = simplify(ret_dict)
            except Exception as e:
                logger.exception("Exception during samplify process.")
                if hasattr(e, 'message'):
                    exc_desc = e.message
                else:
                    exc_desc = str(type(e).__name__) + " : " + str(e.args)
                ret_dict = {"qc": 4, "exception": exc_desc}

        file_uuid = str(uuid.uuid4().hex)
        ret_dict['file_uuid'] = file_uuid

        with self.inst_lock:
            self._pages_list[page] = ret_dict
            if self.do_not_mark_second:
                final_pages = self._pages_list
            else:
                final_pages: list = mark_second_files(self._pages_list)

        message = {"status": "processing",
                   "progress": self._pr(page, 100),
                   "pages": final_pages}

        jstr = json.dumps(message, ensure_ascii=False, indent=4)
        if self._connection:
            self._connection.set(self._id_uuid, jstr, ex=redis_resp_expire)
        return file_uuid

    def final_ready(self):
        with self.inst_lock:
            if self.do_not_mark_second:
                final_pages = self._pages_list
            else:
                final_pages: list = mark_second_files(self._pages_list)

        message = {
            "status": "ready",
            "pages": final_pages,
            "time_seconds": round(time.time() - self.time, 2)
        }

        jstr = json.dumps(message, ensure_ascii=False, indent=4)
        if self._connection:
            self._connection.set(self._id_uuid, jstr, ex=redis_resp_expire)
        else:  # for test
            return self._id_uuid, message

    def clear(self):
        try:
            self._connection.close()
        except:  # noqa
            pass
        # cleare old instances
        t_now = time.time()
        with OutputToRedis.instances_lock:
            # for k, v in OutputToRedis.instances.items()[:]:
            #     _, dt = v
            #     if (dt_now - dt).seconds > (REDIS_EXPIRE * 2):
            #         del OutputToRedis.instances[k]
            OutputToRedis.instances = \
                dict((k, v) for k, v in OutputToRedis.instances.items()
                     if round(t_now - v.time) <= (redis_resp_expire * 2))


# debug
if __name__ == '__main__':
    import threading
    import time

    uuid_test = '123'

    class Conn:
        @staticmethod
        def set(a,b,ex):
            # print(b)
            print(json.loads(b))


    def thread1(page):
        class Obj:
            request_uuid = uuid_test
        time.sleep(page)
        Obj.page_num = page
        OutputToRedis.progress(Obj, 10)
        time.sleep(page)
        OutputToRedis.progress(Obj, 20)
        time.sleep(page)
        OutputToRedis.progress(Obj, 100)


    redis_out = OutputToRedis(uuid_test, Conn)
    c = 3
    redis_out.after_split(c)

    th_fun = [thread1 for x in range(c)]

    threads = []

    for i in range(c):
        th1 = threading.Thread(target=th_fun[i], args=(i,))
        th1.start()
        threads.append(th1)

    for th in threads:
        th.join()

    OutputToRedis.page_completed(redis_out, 0, {"qc": 0, "document": 0}, simplify_flag=True)
    OutputToRedis.page_completed(redis_out, 1, {"qc": 0, "document": 0}, simplify_flag=False)
    OutputToRedis.page_completed(redis_out, 2, {"qc": 0, "document": 0}, simplify_flag=False)