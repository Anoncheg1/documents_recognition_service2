var spec =
{
  "swagger": "2.0",
  "info": {
    "description": "Document recognition service. <p>Statistic [https://opencv.dev.norma.rnb.com](https://opencv.dev.norma.rnb.com). <p> <p>You can find out more  at [https://gitlab.rusnarbank.ru/RNB/opencv](https://gitlab.rusnarbank.ru/RNB/opencv).",
    "version": "1.0.2",
    "title": "opencv",
    "contact": {
      "email": "Chepelev_VS@bel-rusnarbank.ru"
    }
  },
  "tags": [
    {
      "name": "file",
      "description": "Upload file"
    },
    {
      "name": "status",
      "description": "To get result and status"
    }
  ],
  "paths": {
    "/upload": {
      "post": {
        "tags": [
          "file"
        ],
        "summary": "Upload file for detailed recognition",
        "description": "Accepted files: pdf, jpeg, jpg, png",
        "operationId": "uploadFile",
        "consumes": [
          "multipart/form-data"
        ],
        "produces": [
          "application/json"
        ],
        "parameters": [
          {
            "in": "formData",
            "name": "file",
            "description": "Pdf with images or single image for recognition",
            "required": true,
            "type": "file"
          },
          {
            "in": "header",
            "name": "priority",
            "description": "1 is higher, -1 is lower",
            "required": false,
            "type": "integer",
            "default": 0
          }
        ],
        "responses": {
          "200": {
            "description": "File accepted",
            "schema": {
              "$ref": "#/definitions/Id"
            }
          },
          "400": {
            "description": "Invalid input",
            "schema": {
              "$ref": "#/definitions/Exception"
            }
          }
        }
      }
    },
    "/simple_api/upload": {
      "post": {
        "tags": [
          "file"
        ],
        "summary": "Upload file for recognition",
        "description": "Accepted files: pdf, jpeg, jpg, png",
        "operationId": "uploadFileSimple",
        "consumes": [
          "multipart/form-data"
        ],
        "produces": [
          "application/json"
        ],
        "parameters": [
          {
            "in": "formData",
            "name": "file",
            "description": "Pdf with images or single image for recognition",
            "required": true,
            "type": "file"
          },
          {
            "in": "header",
            "name": "priority",
            "description": "1 is higher, -1 is lower",
            "required": false,
            "type": "integer",
            "default": 0
          }
        ],
        "responses": {
          "200": {
            "description": "File accepted",
            "schema": {
              "$ref": "#/definitions/Id"
            }
          },
          "400": {
            "description": "Invalid input",
            "schema": {
              "$ref": "#/definitions/Exception"
            }
          }
        }
      }
    },
    "/simple_api/passport_upload": {
      "post": {
        "tags": [
          "file"
        ],
        "summary": "Upload passport for fast recognition",
        "description": "Accepted files: pdf, jpeg, jpg, png",
        "operationId": "uploadFileSimpleFastP",
        "consumes": [
          "multipart/form-data"
        ],
        "produces": [
          "application/json"
        ],
        "parameters": [
          {
            "in": "formData",
            "name": "file",
            "description": "Pdf with images or single image for recognition",
            "required": true,
            "type": "file"
          },
          {
            "in": "header",
            "name": "priority",
            "description": "1 is higher, -1 is lower",
            "required": false,
            "type": "integer",
            "default": 0
          }
        ],
        "responses": {
          "200": {
            "description": "File accepted",
            "schema": {
              "$ref": "#/definitions/Id"
            }
          },
          "400": {
            "description": "Invalid input",
            "schema": {
              "$ref": "#/definitions/Exception"
            }
          }
        }
      }
    },
    "/simple_api/driving_license_upload": {
      "post": {
        "tags": [
          "file"
        ],
        "summary": "Upload driving license for fast recognition",
        "description": "Accepted files: pdf, jpeg, jpg, png",
        "operationId": "uploadFileSimpleFast",
        "consumes": [
          "multipart/form-data"
        ],
        "produces": [
          "application/json"
        ],
        "parameters": [
          {
            "in": "formData",
            "name": "file",
            "description": "Pdf with images or single image for recognition",
            "required": true,
            "type": "file"
          },
          {
            "in": "header",
            "name": "priority",
            "description": "1 is higher, -1 is lower",
            "required": false,
            "type": "integer",
            "default": 0
          }
        ],
        "responses": {
          "200": {
            "description": "File accepted",
            "schema": {
              "$ref": "#/definitions/Id"
            }
          },
          "400": {
            "description": "Invalid input",
            "schema": {
              "$ref": "#/definitions/Exception"
            }
          }
        }
      }
    },
    "/simple_api/passp_and_dlic_upload": {
      "post": {
        "tags": [
          "file"
        ],
        "summary": "Upload passport and driving license for fast recognition",
        "description": "Accepted files: pdf, jpeg, jpg, png",
        "operationId": "uploadFileSimpleFastpdl",
        "consumes": [
          "multipart/form-data"
        ],
        "produces": [
          "application/json"
        ],
        "parameters": [
          {
            "in": "formData",
            "name": "file",
            "description": "Pdf with images or single image for recognition",
            "required": true,
            "type": "file"
          },
          {
            "in": "header",
            "name": "priority",
            "description": "1 is higher, -1 is lower",
            "required": false,
            "type": "integer",
            "default": 0
          }
        ],
        "responses": {
          "200": {
            "description": "File accepted",
            "schema": {
              "$ref": "#/definitions/Id"
            }
          },
          "400": {
            "description": "Invalid input",
            "schema": {
              "$ref": "#/definitions/Exception"
            }
          }
        }
      }
    },
    "/simple_api/barcodes_only_upload": {
      "post": {
        "tags": [
          "file"
        ],
        "summary": "Upload pages with barcodes for fast recognition",
        "description": "Accepted files: pdf, jpeg, jpg, png",
        "operationId": "uploadFileSimpleFastb",
        "consumes": [
          "multipart/form-data"
        ],
        "produces": [
          "application/json"
        ],
        "parameters": [
          {
            "in": "formData",
            "name": "file",
            "description": "Pdf with images or single image for recognition",
            "required": true,
            "type": "file"
          },
          {
            "in": "header",
            "name": "priority",
            "description": "1 is higher, -1 is lower",
            "required": false,
            "type": "integer",
            "default": 0
          }
        ],
        "responses": {
          "200": {
            "description": "File accepted",
            "schema": {
              "$ref": "#/definitions/Id"
            }
          },
          "400": {
            "description": "Invalid input",
            "schema": {
              "$ref": "#/definitions/Exception"
            }
          }
        }
      }
    },
    "/get": {
      "get": {
        "tags": [
          "status"
        ],
        "summary": "Status and result",
        "description": "Returnes status and result for uploaded file by id",
        "operationId": "getStatus",
        "produces": [
          "application/json"
        ],
        "parameters": [
          {
            "name": "id",
            "in": "query",
            "description": "ID of uploaded file",
            "required": true,
            "type": "string",
            "pattern": "^[a-f0-9]{32}$"
          }
        ],
        "responses": {
          "200": {
            "description": "Status with or without result",
            "schema": {
              "$ref": "#/definitions/Response"
            }
          },
          "404": {
            "description": "Invalid key or the key is expired",
            "schema": {
              "$ref": "#/definitions/Exception"
            }
          }
        }
      }
    },
    "/pdf_pages/get": {
      "get": {
        "tags": [
          "status"
        ],
        "summary": "Pages of PDF file",
        "description": "Returnes PDF page of PDF by file_uuid",
        "operationId": "getPage",
        "produces": [
          "application/pdf"
        ],
        "parameters": [
          {
            "name": "id",
            "in": "query",
            "description": "file_uuid of page",
            "required": true,
            "type": "string",
            "pattern": "^[a-f0-9]{32}$"
          }
        ],
        "responses": {
          "200": {
            "description": "file"
          },
          "404": {
            "description": "Invalid key or the key is expired",
            "schema": {
              "$ref": "#/definitions/Exception"
            }
          }
        }
      }
    }
  },
  "definitions": {
    "Id": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "pattern": "^[a-f0-9]{32}$"
        }
      },
      "xml": {
        "name": "Id"
      }
    },
    "Exception": {
      "type": "object",
      "properties": {
        "status": {
          "type": "string",
          "example": "exception"
        },
        "description": {
          "type": "string"
        }
      },
      "xml": {
        "name": "Exception"
      }
    },
    "Response": {
      "type": "object",
      "properties": {
        "status": {
          "type": "string",
          "example": "ready"
        },
        "pages": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "document_type": {
                "type": "integer",
                "format": "int64"
              },
              "qc": {
                "type": "integer",
                "format": "int64"
              },
              "file_uuid": {
                "type": "string",
                "pattern": "^[a-f0-9]{32}$"
              }
            }
          }
        }
      },
      "xml": {
        "name": "Response"
      }
    },
    "Response140": {
      "type": "object",
      "description": "Passport main page and Driving license",
      "properties": {
        "status": {
          "type": "string",
          "example": "ready"
        },
        "pages": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "document_type": {
                "type": "integer",
                "format": "int64",
                "example": 140
              },
              "description": {
                "type": "string",
                "example": "Passport main page and Driving license back side"
              },
              "qc": {
                "type": "integer",
                "format": "int64"
              },
              "passport_main": {
                "type": "object",
                "properties": {
                  "qc": {
                    "type": "integer",
                    "format": "int64"
                  },
                  "MRZ": {
                    "example": null,
                    "description": "may be null"
                  },
                  "main_top_page": {
                    "example": null,
                    "description": "may be null"
                  },
                  "main_bottom_page": {
                    "type": "object",
                    "example": null,
                    "description": "may be null"
                  },
                  "serial_number": {
                    "type": "string",
                    "example": null,
                    "pattern": "^[0-9]{10}$",
                    "description": "Sirial number and passport number"
                  },
                  "serial_number_check": {
                    "type": "boolean",
                    "example": false,
                    "description": "The result of comparision with mrz.s_number"
                  },
                  "suggest": {
                    "type": "object",
                    "example": null,
                    "description": "may be null"
                  }
                }
              },
              "driving_license": {
                "type": "object",
                "properties": {
                  "qc": {
                    "type": "integer",
                    "format": "int64"
                  },
                  "side": {
                    "type": "string",
                    "example": null,
                    "pattern": "^(front|back)$",
                    "description": "May be null."
                  }
                }
              }
            }
          }
        }
      }
    },
    "Response100": {
      "type": "object",
      "description": "Passport main page",
      "properties": {
        "status": {
          "type": "string",
          "example": "ready"
        },
        "pages": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "document_type": {
                "type": "integer",
                "format": "int64",
                "example": 100
              },
              "description": {
                "type": "string",
                "example": "Passport main page"
              },
              "qc": {
                "type": "integer",
                "format": "int64"
              },
              "MRZ": {
                "example": null,
                "description": "may be null"
              },
              "main_top_page": {
                "type": "object",
                "properties": {
                  "vidan": {
                    "type": "string",
                    "example": null
                  },
                  "data_vid": {
                    "type": "string",
                    "example": "2017",
                    "description": "YYYY or DD.MM.YYYY"
                  },
                  "code_pod": {
                    "type": "string",
                    "example": "160-032"
                  }
                }
              },
              "main_bottom_page": {
                "type": "object",
                "properties": {
                  "F": {
                    "type": "string",
                    "example": null
                  },
                  "I": {
                    "type": "string",
                    "example": null
                  },
                  "O": {
                    "type": "string",
                    "example": null
                  },
                  "gender": {
                    "type": "string",
                    "example": null,
                    "pattern": "^(МУЖ|ЖЕН)$"
                  },
                  "birth_date": {
                    "type": "string",
                    "example": "2017",
                    "description": "YYYY or DD.MM.YYYY"
                  },
                  "birth_place": {
                    "type": "string",
                    "example": null
                  }
                }
              },
              "serial_number": {
                "type": "string",
                "example": null
              },
              "serial_number_check": {
                "type": "boolean",
                "example": false,
                "description": "The result of comparision with mrz.s_number"
              },
              "suggest": {
                "type": "object",
                "example": null,
                "description": "may be null"
              }
            }
          }
        }
      }
    },
    "Response130_back": {
      "type": "object",
      "description": "Driving license back side",
      "properties": {
        "status": {
          "type": "string",
          "example": "ready"
        },
        "pages": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "document_type": {
                "type": "integer",
                "format": "int64",
                "example": 130
              },
              "description": {
                "type": "string",
                "example": "Driving license and Passport not main page"
              },
              "qc": {
                "type": "integer",
                "format": "int64"
              },
              "side": {
                "type": "string",
                "example": "back",
                "pattern": "^(front|back)$"
              },
              "s_number": {
                "type": "string",
                "example": "5221404468",
                "pattern": "^[0-9]{10}$",
                "description": "may be null"
              }
            }
          }
        }
      }
    },
    "Response130_front": {
      "type": "object",
      "description": "Driving license front side",
      "properties": {
        "status": {
          "type": "string",
          "example": "ready"
        },
        "pages": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "document_type": {
                "type": "integer",
                "format": "int64",
                "example": 130
              },
              "description": {
                "type": "string",
                "example": "Driving license"
              },
              "qc": {
                "type": "integer",
                "format": "int64",
                "example": "5221404468",
                "pattern": "^[0-9]{10}$"
              },
              "side": {
                "type": "string",
                "example": "front",
                "pattern": "^(front|back)$"
              },
              "fam_rus": {
                "type": "string"
              },
              "fam_eng": {
                "type": "string"
              },
              "fam_check": {
                "type": "boolean",
                "description": "The result of comparision rus and eng"
              },
              "name_rus": {
                "type": "string"
              },
              "name_eng": {
                "type": "string"
              },
              "name_check": {
                "type": "boolean",
                "description": "The result of comparision rus and eng"
              },
              "p3": {
                "type": "string",
                "description": "YYYY or DD.MM.YYYY"
              },
              "birthplace3_rus": {
                "type": "string"
              },
              "birthplace3_eng": {
                "type": "string"
              },
              "birthplace3_check": {
                "type": "boolean"
              },
              "p4a": {
                "type": "string",
                "description": "YYYY or DD.MM.YYYY"
              },
              "p4b": {
                "type": "string",
                "description": "YYYY or DD.MM.YYYY"
              },
              "p4ab_check": {
                "description": "09.10.2018 == 09.10.2028 - True; 2018 == 2028 - True"
              },
              "p4c_rus": {
                "type": "string"
              },
              "p4c_eng": {
                "type": "string"
              },
              "p4c_check": {
                "type": "boolean"
              },
              "p5": {
                "type": "string",
                "pattern": "$[0-9]{10}",
                "description": "Serial number."
              },
              "p8_rus": {
                "type": "string"
              },
              "p8_eng": {
                "type": "string"
              },
              "p8_check": {
                "type": "boolean"
              },
              "suggest": {
                "type": "object",
                "example": null,
                "description": "may be null"
              },
              "categories": {
                "type": "array",
                "items": {
                  "type": "string",
                  "example": [
                    "C",
                    "C1",
                    "B",
                    "B1"
                  ],
                  "description": "we have problem with [M] for now"
                }
              }
            }
          }
        }
      }
    },
    "MRZ": {
      "type": "object",
      "description": "Passport Machine-readable zone",
      "properties": {
        "s_number": {
          "type": "string",
          "example": "9214844670",
          "pattern": "^[0-9]{10}$",
          "description": "For: 9214 N844670"
        },
        "s_number_check": {
          "type": "boolean",
          "example": true
        },
        "birth_date": {
          "type": "string",
          "example": "680804",
          "description": "YYMMDD birth date"
        },
        "birth_date_check": {
          "type": "boolean",
          "example": true
        },
        "issue_date": {
          "type": "string",
          "example": "150303",
          "description": "YYMMDD format"
        },
        "code": {
          "type": "string",
          "example": "150017",
          "pattern": "^[0-9]{6}$"
        },
        "issue_date_and_code_check": {
          "type": "boolean",
          "example": true
        },
        "finalok": {
          "type": "boolean",
          "example": true,
          "description": "For s_number, birth_date, issue_date, code"
        },
        "mrz_f": {
          "type": "string",
          "example": "КАДЫРБЕК УУЛУ",
          "description": "surname - may have"
        },
        "mrz_i": {
          "type": "string",
          "example": "БЕКБОЛОТ",
          "description": "first name"
        },
        "mrz_o": {
          "type": "string",
          "example": null,
          "description": "patronymic"
        },
        "mrz_f_check": {
          "type": "boolean",
          "example": false,
          "description": "Compared with main_bottom_page.F"
        },
        "mrz_i_check": {
          "type": "boolean",
          "example": false,
          "description": "Compared with main_bottom_page.I"
        },
        "mrz_o_check": {
          "type": "boolean",
          "example": false,
          "description": "Compared with main_bottom_page.O"
        },
        "gender": {
          "type": "string",
          "pattern": "^(M|F)$",
          "example": "F"
        },
        "gender_check": {
          "type": "boolean",
          "example": false,
          "description": "Compared with main_bottom_page.gender"
        }
      }
    },
    "suggest": {
      "type": "object",
      "description": "closest in writing and poppularity",
      "properties": {
        "F": {
          "type": "string",
          "example": "Юрко",
          "pattern": "[А-Я -]",
          "description": "Surname. May be null."
        },
        "F_gender": {
          "type": "string",
          "example": "UNKNOWN",
          "pattern": "(MALE|FEMALE|UNKNOWN)",
          "description": "May be null."
        },
        "F_score": {
          "type": "number",
          "example": 1,
          "description": "0-1  1:full match."
        },
        "I": {
          "type": "string",
          "example": "Зоя",
          "pattern": "[А-Я -]",
          "description": "First Name. May be null."
        },
        "I_gender": {
          "type": "string",
          "example": "FEMALE",
          "pattern": "(MALE|FEMALE)",
          "description": "May be null."
        },
        "I_score": {
          "type": "number",
          "example": 0.75,
          "description": "0-1  1:full match."
        },
        "O": {
          "type": "string",
          "example": "АЛИ КАЗИ",
          "pattern": "[А-Я -]",
          "description": "Patronymic. May be null."
        },
        "O_gender": {
          "type": "string",
          "example": "FEMALE",
          "pattern": "(MALE|FEMALE)",
          "description": "May be null."
        },
        "O_score": {
          "type": "number",
          "example": 0,
          "description": "0-1  1:full match."
        }
      }
    }
  }
}
