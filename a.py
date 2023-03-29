from flask import Flask
# print(__name__)
app = Flask(__name__, template_folder='./',
            static_url_path='/static',
            static_folder='/home/u/sources/documents_recognition_service/docker/worker/code/test'
            )

from flask import render_template
from flask import abort, redirect, url_for
from flask import request
from werkzeug.utils import secure_filename
from flask import json
import uuid
import logging
import os
import requests
import json
ct_json = {'Content-Type': 'application/json'}

@app.route("/")
def hell():
    return render_template('a.html')

# mock
@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' in request.files:
            uuidstr = str(uuid.uuid4().hex)
            return json.dumps({"id": uuidstr}), 200, ct_json
        else:
            return 'bad file {secure_filename(f.filename)}'
    else:
        return "bad"

# mock
@app.route('/get', methods=['GET'])
def get():
    """
    Метод для получения результата обработки документов по jod id

    JOB ID выдаётся методом api_root
    - Получает JOB ID с помощью параметра id
    - Обращается к Redis с JOB ID
    - Если такой ключ есть, то отдается значение этого ключа
    """
    app.logger.debug('t' + str(type(request.args)))

    rid = request.args.get('id', '')
    print('get', str(type(rid)), rid, rid == '123', len(rid))
    if rid == "":
        return json.dumps({"status": "exception", "description": "key not specified"}), 400, ct_json
    if rid == '123' or len(str(rid)) == 32 :
        # print(rid)
        return {'status':'processing'}, 200, ct_json
    print('tf')
    return json.dumps({"status": "exception", "description": "key not exists"}), 404, ct_json


@app.route("/upload_passport_test/<filename>", methods=['GET'])
def upload_passport_test(filename: str):
    # logging.error('app.static_folder '+ app.static_folder)
    # logging.error('app.static_url_path '+ app.static_url_path)
    # logging.error('app.template_folder ' + app.template_folder)
    # logging.error(f'file {filename}')
    # app.logger.debug(os.path.exists(os.path.join(app.static_folder, filename)))
    # app.logger.debug(os.path.exists(os.path.join(app.template_folder, 'a.html')))
    filepath=os.path.join(app.static_folder, filename)
    if os.path.exists(filepath):
        # -- upload
        request.files = {'file': open(filepath, 'rb')}
        request.method = 'POST'
        r = upload()
        rid = json.loads(r[0])['id']
        # request.args = {'id': rid}
        # r = get()
        # app.logger.debug("rasds " + json.dumps(r[0], indent=4))
        # app.logger.debug("rasds " + json.dumps(json.loads(str(r[0])), indent=4))
        return redirect('/get?id='+rid)
        # return json.dumps(r[0], indent=4), 200
    # app.logger.debug("rasds " + json.dumps(r[0], indent=4)), 200, 'text'
    else:
        return 'file do not exist', 404
    # if 'file' in request.files:
    #     uuidstr = str(uuid.uuid4().hex)
    #     return json.dumps({"id": uuidstr}), 200, ct_json
    # else:
    #     return 'bad file {secure_filename(f.filename)}'


# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"

if __name__ == "__main__":
    print("start")
    app.run(host='localhost', port=8080, debug=True)
