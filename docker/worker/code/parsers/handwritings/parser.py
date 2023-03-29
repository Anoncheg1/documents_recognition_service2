import numpy as np
# own
from parsers.handwritings.static_templated_handwritings import parse_handwritings_static
from parsers.handwritings.variable_document_handwritings import parse_variable_fields
from utils.doctype_by_text import DocTypes

# DOCT_TYPES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# undertypes = (None, 10, None, 45, None, None,  42, None, 40)  # 2, 4, 7 ,9
SUPPORTED_DOCT_TYPES = {
    # 1,
    2: DocTypes.consentToProcessingOfPersonalData,
    3: DocTypes.individualConditions,
    4: DocTypes.loginAssignmentNotice,
    # 5:applicationForOnlineServices,
    6: DocTypes.applicationForAutocredit,
    7: DocTypes.applicationForAdvanceAcceptance,
    8: DocTypes.applicationForTransferringFromAccountFL,
    9: DocTypes.borrowerProfile}

REVERSE_SUPPORTED_DOCT_TYPES = {}
for k, v in SUPPORTED_DOCT_TYPES.items():
    assert k not in REVERSE_SUPPORTED_DOCT_TYPES
    REVERSE_SUPPORTED_DOCT_TYPES[v] = k


def parser_handwritings(img: np.ndarray, doc_type: int, page: int = 1) -> dict or None:
    assert doc_type in REVERSE_SUPPORTED_DOCT_TYPES.keys()

    doc_type_here = REVERSE_SUPPORTED_DOCT_TYPES[doc_type]

    ret_list = None
    if doc_type_here in [2, 4, 7, 9]:
        ret_list = parse_handwritings_static(img, doc_type_here, page)
    elif doc_type_here in [3, 6, 8]:
        ret_list = parse_variable_fields(img, doc_type_here, page)

    return ret_list
