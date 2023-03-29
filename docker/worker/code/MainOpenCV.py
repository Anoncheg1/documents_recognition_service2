"""
    Основной модуль программы для распознавания PDF документов
"""
import os
import cv2 as cv
import redis
import inject
import tempfile
from multiprocessing.pool import Pool
from PIL import Image
# own
from logger import logger as log, set_uuid
from groonga import FIOChecker
from predict_utils.predict import predict
from predict_utils.predict import predict_is_text
from predict_utils.predict import predict_is_handwrited
from utils.progress_and_output import OutputToRedis, PROFILING
from utils.pdf_convertors import pdf2png

single_instance = FIOChecker(10)
# inject heavy external dependencies for testing
inject.configure(lambda binder: binder
                 .bind('FIOChecker', single_instance)
                 .bind_to_provider('predict', lambda: predict)
                 .bind_to_provider('predict_is_text', lambda: predict_is_text)
                 .bind_to_provider('predict_is_handwrited', lambda: predict_is_handwrited))
# own
from doc_types import DocTypeDetected
from parsers.parser import method_number_list
from config import redis_host, redis_port
from config import redis_resp_expire

PAGE_THREADS = 2


def parser_call(obj: DocTypeDetected) -> dict:
    """
    Каждый метод - каждый тип документа. Вызывает методы парсинга на основе типа распознанного документа.

    :param obj: side effect! doc_type, description
    :return:
    """
    try:
        return method_number_list[obj.doc_type](obj).OUTPUT_OBJ
    except Exception as e:
        log.exception("Uncatched exception in ParserClass")
        if hasattr(e, 'message'):
            return {"qc": 4, "exception": "ParserClass " + e.message, "traceback": str(e)}
        else:
            return {"qc": 4, "exception": "ParserClass " + str(type(e).__name__) + " : " + str(e.args)}


def page_processing(procnum: int, file_path: str, request_uuid: str,
                    doc_classes: list, barcodes_only: bool) -> (int, dict):
    """
    Метод обработки страницы
    Вызывается асинхронно для каждой отдельной страницы

    - Принимает номер процесса procnum
    - Принимает словарь для записи полученных значений return_dict
    - Принимает имя файла .png с одной страницей pdf-документа file_name
    """
    bad_image = (procnum, {"qc": 4, "exception": "Bad image file."})
    try:
        try:
            # TODO: check file_name at the disk without crash risk
            if not os.path.exists(file_path) or os.stat(file_path).st_size <= 0:
                return bad_image
            image = cv.imread(file_path)  # may crash even without exception
            if image is None:
                return bad_image
        except:
            return bad_image
        # 1) detect doc type
        obj = DocTypeDetected(request_uuid, procnum)  # (image, doc_classes, request_uuid, procnum)
        if barcodes_only:
            obj.detect_barcodes(image)  # doc_type =0 or >0
        elif doc_classes is not None and len(doc_classes) > 0:  # not text docs
            obj.detect_not_text_docs(image, doc_classes)
        else:
            obj.detect_doctype(image)  # universal

        OutputToRedis.progress(obj, 1, obj.doc_type)
        if barcodes_only:
            ret = {'qc': 0}
        else:
            # 2) parse
            ret: dict = parser_call(obj)  # side effect on obj.doctype, obj.description, if doctype=0, qc=4
        # TODO: retry if universal parser return bad passport
        # set our fields in the beginning of the dictionary
        outdict = {'document_type': obj.doc_type}
        if obj.description is not None:
            outdict['description'] = obj.description
        if obj.text_doc_orientation is not None:
            outdict['text_doc_orientation'] = obj.text_doc_orientation
        outdict.update(ret)
        # page
        if obj.page is not None:
            outdict["page"] = obj.page
    except Exception as e:
        log.exception("Uncatched exception in page_processing" + str(e.args))
        raise
    return procnum, outdict


class MainProcessingClass:
    """
    Основной класс, с которого начинается обработка изображения
    WORKER в очереди opencv-tasks обращается именно к нему
    """

    def __init__(self, id_processing, simple_flag=False, doc_classes=None, barcodes_only=False):
        """

        :param id_processing: redis job uuid in queue.
        JOB ID для записи статуса и результатов выполнения по этому ключу в Redis
        :param simple_flag: simplified response
        :param doc_classes: limit doc_type which will be parsed
        """
        # Declaration block --
        self.doc_classes = [] if doc_classes is None else doc_classes
        self.id_processing = str(id_processing)
        self.simple_flag = simple_flag
        self.barcodes_only = barcodes_only
        self.redis_out = None
        self.redis_page_files = None
        set_uuid(id_processing)  # add uuid to logger

    def run(self, file_base64, name: str):
        """
        Cannot be used in tests
        :param file_base64:
        :param name:
        :return:
        """
        with tempfile.TemporaryDirectory() as tempdirpath:
            try:
                filepath = os.path.join(tempdirpath, name)  # self.id_processing + name)  # declare

                with open(filepath, "wb") as fh:
                    fh.write(file_base64)

                # result's output and source file output
                with redis.Redis(host=redis_host, port=redis_port, db=1) as conn, \
                        redis.Redis(host=redis_host, port=redis_port, db=6) as page_files_conn:
                    self.redis_page_files = page_files_conn  # FOR FILES
                    self.workflow(filepath, conn)  # MAIN SUBROUTINE
            except Exception as e:
                log.exception("Exception in MainProcessingClass.run")
                raise e

    def multithread_processing(self, filelist) -> list:
        """ Multithread page processing

        self.doc_classes: recognize as one of these classes

        self.redis_out: if not None

        :param filelist: # файлы PNG страниц PDF входящего файла
        :return: [procnum:(procnum, new_obj.OUTPUT_OBJ), ....] - sorted pages FOR DEBUG
        """
        ret_list: list = [None for _ in range(len(filelist))]  # for debug
        if PROFILING:  # cannot go through threads
            for i, fp in enumerate(filelist):
                inx, dict_list = page_processing(i, fp, self.id_processing, self.doc_classes, self.barcodes_only)
                if self.redis_out:
                    self.redis_out.page_completed(inx, dict_list, self.simple_flag)
                ret_list[inx] = dict_list  # for test
        else:
            # Processes
            def callback_result(result):
                inx, dict_list = result
                # progress and send file to redis
                if self.redis_out:
                    file_uuid = self.redis_out.page_completed(inx, dict_list, self.simple_flag)
                    if self.redis_page_files:
                        pdf_file_path = os.path.splitext(filelist[inx])[0] + '.pdf'
                        Image.open(filelist[inx]).convert('RGB').save(pdf_file_path)
                        with open(pdf_file_path, "rb") as handler_f:
                            self.redis_page_files.set(file_uuid, handler_f.read(), ex=redis_resp_expire)
                ret_list[inx] = dict_list  # for test
            # Pool
            executor = Pool(processes=PAGE_THREADS)  # clear leaked memory with process death
            for i, fp in enumerate(filelist):
                executor.apply_async(
                    page_processing, args=(i, fp, self.id_processing, self.doc_classes, self.barcodes_only),
                    callback=callback_result)
            executor.close()
            executor.join()

        return ret_list  # for test

    @staticmethod
    def split(filepatch):
        ext_fname = os.path.splitext(filepatch)[1].lower()
        if ext_fname == '.pdf':
            filelist = pdf2png(filepatch, tmpdir=os.path.split(filepatch)[0])  # split
        else:
            filelist = [filepatch]
        return filelist

    def workflow(self, filepath: str, conn: redis.Redis = None):
        """ - split pdf file
            - calling main_processing(),
            - send response and status: "processing"
            Using:
            - self.filepatch
        """
        try:
            self.redis_out = OutputToRedis(self.id_processing, conn, do_not_mark_second=self.barcodes_only)  # instantiation
            # 1) SPLIT
            filelist = MainProcessingClass.split(filepath)
            self.redis_out.after_split(len(filelist))
            # 2) PROCESS
            self.multithread_processing(filelist)  # result in self.redis_out
            self.test = self.redis_out.final_ready()  # noqa
            self.redis_out.final_ready()
        except Exception as e:
            if hasattr(e, 'message'):
                exc_desc = e.message
            else:
                exc_desc = str(type(e).__name__) + " : " + str(e.args)
            self.redis_out.exception(exc_desc)
            log.exception(exc_desc)

        self.redis_out.clear()
        try:
            self.redis_page_files.close()
        except:  # noqa
            pass
        log.info("File " + str(filepath) + " successfully processed.")
        if hasattr(self, 'test'):
            return self.test  # self.test


def pages_recognition(file_base64, name, id_processing):
    """ External call """
    MainProcessingClass(id_processing).run(file_base64, name)


def pages_recognition_simple(file_base64, name: str, id_processing):
    """ External call """
    MainProcessingClass(id_processing, simple_flag=True).run(file_base64, name)


def passport_upload(file_base64, name: str, id_processing):
    """ External call """
    MainProcessingClass(id_processing, simple_flag=True, doc_classes=[100]).run(file_base64, name)


def driving_license_upload(file_base64, name: str, id_processing):
    """ External call """
    MainProcessingClass(id_processing, simple_flag=True, doc_classes=[130]).run(file_base64, name)


def passp_and_dlic_upload(file_base64, name: str, id_processing):
    """ External call """
    MainProcessingClass(id_processing, simple_flag=True, doc_classes=[100, 130, 140]).run(file_base64, name)


def barcodes_only_upload(file_base64, name: str, id_processing):
    """ External call """
    MainProcessingClass(id_processing, simple_flag=True, barcodes_only=True).run(file_base64, name)
