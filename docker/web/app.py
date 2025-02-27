"""
    Веб-сервер с ипользованием Flask API для взаимодействия со скриптом распознавания
"""

import json
import os
from redis import Redis
import rq
import uuid
from flask import Flask, request, Response, send_from_directory
from werkzeug.utils import secure_filename
from waitress import serve
import time
import rq_dashboard

from config import redis_host, redis_port
from config import redis_get_request_socket_timeout
from config import redis_resp_expire

app = Flask(__name__)
app.config.from_object(rq_dashboard.default_settings)
app.register_blueprint(rq_dashboard.blueprint, url_prefix="/rq")
app.config["RQ_DASHBOARD_REDIS_URL"] = "redis://" + redis_host + ":" + redis_port + "/0"

job_timeout = 60 * 60 * 3
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpeg', 'jpg', 'jpe'}
ct_json = {'Content-Type': 'application/json'}
ct_pdf = {'Content-Type': 'application/pdf'}
ct_png = {'Content-Type': 'image/png'}
# Подключение для получения результатов обработки документов
redis_json = Redis(host=redis_host, port=redis_port, db=1)
# Подключение для получения страниц PDF из результата обработки документов
redis_page_files = Redis(host=redis_host, port=redis_port, db=6)
# Подключение для создания очереди в Redis с помощью python-rq
conn = Redis(host=redis_host, port=redis_port, db=0, retry_on_timeout=True,
             socket_timeout=redis_get_request_socket_timeout)
queue_medium = rq.Queue('in_medium',
                        connection=conn,
                        default_timeout=job_timeout)
queue_high = rq.Queue('in_high',
                      connection=conn,
                      default_timeout=job_timeout)


def allowed_file(filename):
    ext = os.path.splitext(filename)
    # latin and unicode
    return (ext[1][1:].lower() in ALLOWED_EXTENSIONS) or (ext[1] == '' and ext[0].lower() in ALLOWED_EXTENSIONS)


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/spec.js')
def spec():
    return app.send_static_file('spec.js')


@app.route('/dist/<file>')
def send_js(file):
    return send_from_directory('static/dist', file)


@app.route('/get', methods=['GET'])
def get_result():
    """
    Метод для получения результата обработки документов по jod id
    
    JOB ID выдаётся методом api_root
    - Получает JOB ID с помощью параметра id
    - Обращается к Redis с JOB ID
    - Если такой ключ есть, то отдается значение этого ключа
    """
    rid = request.args.get('id', '')
    if rid == "":
        return json.dumps({"status": "exception", "description": "key not specified"}), 400, ct_json
    if redis_json.exists(rid) != 0:
        return redis_json.get(rid), 200, ct_json
    return json.dumps({"status": "exception", "description": "key not exists"}), 404, ct_json


@app.route('/pdf_pages/get', methods=['GET'])
def get_pdf_pages():
    """
    Метод для получения страниц PDF из результата обработки по jod id

    JOB ID вывозвращается в get_result()
    - Получает JOB ID с помощью параметра id
    - Обращается к Redis с JOB ID
    - Если такой ключ есть, то отдается значение этого ключа
    """
    rid = request.args.get('id', '')
    if rid == "":
        return json.dumps({"status": "exception", "description": "key not specified"}), 400, ct_json
    if redis_page_files.exists(rid) != 0:
        return redis_page_files.get(rid), 200, ct_pdf
    return json.dumps({"status": "exception", "description": "key not exists"}), 404, ct_json


# common methods -----
def file_preprocessing(files: list) -> (int, object):
    """ return
    1 - exception
    0 - OK
    """
    if len(files) > 1:
        return 1, (json.dumps(
            {"status": "exception", "description": "Not support more than one file for now"}), 400, ct_json)

    file = files[0]

    if file.filename == "":
        return 1, (json.dumps({"status": "exception", "description": "file not exists"}), 400, ct_json)

    file_name = secure_filename(file.filename)  # ASCII only
    uuidstr = str(uuid.uuid4().hex)

    # replace filename with uuid + last 4 ASCII symbols
    file_name = uuidstr + '.' + file_name[-4:]

    if not file or not allowed_file(file_name):
        return 1, (json.dumps(
            {"status": "exception", "description": "file is not a .pdf .png .jpg .jpeg file"}), 400, ct_json)

    file.save(file_name)
    with open(file_name, "rb") as handler_f:
        f_binary = handler_f.read()
        # f_base64 = base64.b64encode(handler_f.read())
    os.remove(file_name)
    # return 0, (f_base64, file_name, uuidstr)
    return 0, (f_binary, file_name, uuidstr)


def parse(method: str, path: str):  # -> json
    """
    :param method: used for parsing
    :param path: for description only
    :param queue:
    """
    queue = queue_medium

    if request.method == "POST":

        # get optional parameter
        priority = request.headers.get('priority')
        if priority and int(priority) > 0:
            queue = queue_high

        # get file
        files: list = list(request.files.values())
        status, res = file_preprocessing(files)
        if status == 1:  # bad
            return res  # exception message

        f_binary, file_name, uuidstr = res
        # job_timeout - time to between start and stop job (not used)
        # MAIN QUEUE
        queue.enqueue(f=method, args=(f_binary, file_name, uuidstr),
                      description=path, meta=int(time.time()),
                      at_front=True, job_id=uuidstr)  # LIFO

        if redis_json.set(uuidstr, json.dumps({"status": "in pool"}), ex=redis_resp_expire):
            return json.dumps({"id": uuidstr}), 200, ct_json
        else:
            return json.dumps({"status": "exception", "description": "fail to push task to redis"}), 500, ct_json

    elif request.method == 'GET':
        return '''
                <!doctype html>
                <title>Upload new File</title>
                <h1>Upload new File</h1>
                <form action="''' + path + '''" method=post enctype=multipart/form-data>
                <p><input type=file name=file>
                    <input type=submit value=Upload>
                </form>
            ''', 200, {'Content-Type': 'text/html'}


upload = 'upload'
passport_upload = 'passport_upload'
driving_license_upload = 'driving_license_upload'
passp_and_dlic_upload = 'passp_and_dlic_upload'
barcodes_only_upload = 'barcodes_only_upload'


@app.route('/' + upload, methods=['GET', 'POST'])
def parse_detailed():
    """
    Метод для загрузки документов на обработку страниц
    
    - Принимает документ с полем pdf, png, jpg, jpeg
    - Добавляет элемент в очередь queue.enqueue для обработки в MainOpenCV.page_recognition
    - Выставляет для текущего JOB ID в Redis значение "in pool"
    - Отдает JOB ID
    """
    return parse('MainOpenCV.pages_recognition', upload)


@app.route('/simple_api/' + upload, methods=['GET', 'POST'])
def parse2():
    return parse('MainOpenCV.pages_recognition_simple', 'simple_upload')


@app.route('/simple_api/' + passport_upload, methods=['GET', 'POST'])
def parse3():
    return parse('MainOpenCV.' + passport_upload, passport_upload)


@app.route('/simple_api/' + driving_license_upload, methods=['GET', 'POST'])
def parse4():
    return parse('MainOpenCV.' + driving_license_upload, driving_license_upload)


@app.route('/simple_api/' + passp_and_dlic_upload, methods=['GET', 'POST'])
def parse5():
    return parse('MainOpenCV.' + passp_and_dlic_upload, passp_and_dlic_upload)


@app.route('/simple_api/' + barcodes_only_upload, methods=['GET', 'POST'])
def parse6():
    return parse('MainOpenCV.' + barcodes_only_upload, barcodes_only_upload)


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000, threads=10)
    # app.run(host='0.0.0.0', debug=True, use_reloader=False)
