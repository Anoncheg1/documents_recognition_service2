import inject
from predict_utils.cvutils import prepare
import numpy as np
# own
from utils.doctype_by_text import simple_detect_type
from utils.doctype_by_barcode import read_barcode
from utils.progress_and_output import OutputToRedis


class DocTypeDetected:

    def __init__(self, request_uuid: str = None, page_num: int = None):
        # images
        self.original: np.array = None
        self.not_resized_cropped: np.array = None
        # detected document type:
        self.doc_type = 0
        self.description = None
        self.text_doc_orientation = None
        self.page = None
        # for progress output
        self.request_uuid = request_uuid
        self.page_num = page_num

    def detect_barcodes(self, image):
        self.original = image
        self.doc_type, self.description, self.page = read_barcode(image)  # doc_type=0 or >0

    def detect_text_docs(self, image):
        self.doc_type, self.description, self.page = read_barcode(image)  # doc_type=0 or >0
        if self.doc_type == 0:
            text_doc_orientation = OutputToRedis.get_text_doc_orientation(self.request_uuid)
            self.doc_type, self.description, self.text_doc_orientation = \
                simple_detect_type(image, text_doc_orientation)

    def detect_not_text_docs(self, image, doc_classes: list = None):
        self._detect_advanced(image)
        # TODO: if bad recognition - try detect by barcode and text?
        if doc_classes is not None:
            if self.doc_type == 140 and 100 in doc_classes and 140 not in doc_classes:  # only passport
                self.doc_type = 100  # ignore driving license
            elif self.doc_type == 140 and 130 in doc_classes and 140 not in doc_classes:  # only driving license
                self.doc_type = 130  # ignore passport
            elif self.doc_type not in doc_classes:
                self.doc_type = 0

    @inject.params(predict_is_text='predict_is_text')  # param name to a binder key.
    def detect_doctype(self, image, predict_is_text=None):
        self.original = image

        is_text = predict_is_text(image)
        if is_text:  # for text
            self.detect_text_docs(image)
        else:
            # not text
            self.detect_not_text_docs(image)



    @inject.params(predict='predict')  # param name to a binder key.
    def _detect_advanced(self, image, predict=None):
        """
        ret_object;
            100 - not_resized_cropped
            110,120,130 - original image
            140 - original + not_resized_cropped

        :param predict: callable(np.ndarray) -> (int or None, str or None, np.array or None)
        """
        self.original = image
        im_resized, gray_resized_not_cropped_angle_fixed, self.not_resized_cropped = prepare(image.copy(), rate=1)

        cl, subclass, r_rot = predict(im_resized, gray_resized_not_cropped_angle_fixed)
        if r_rot is not None and r_rot != 0:
            for _ in range(r_rot):
                self.not_resized_cropped = np.rot90(self.not_resized_cropped)

        # self.not_resized_cropped = img  # for driving_lic and passport

        if cl == 0:  # passport
            if subclass == 'passport_main':
                self.doc_type = 100
                self.description = 'Passport: main page'
            else:
                self.doc_type = 101
                self.description = 'Passport: not main page'
        elif cl == 1:  # pts
            self.doc_type = 110
            self.description = 'PTS'
        elif cl == 2:  # photo
            self.doc_type = 120
            self.description = 'Photo'
        elif cl == 3:  # vodit_udostav
            self.doc_type = 130
            self.description = 'Driving license'
        else:  # cl == 4:  # vodit_udostav and passport
            if subclass == 'passport_main':
                self.doc_type = 140
                self.description = 'Passport main page and Driving license'
            else:
                self.doc_type = 130
                self.description = 'Driving license and Passport not main page'
