FROM python:3.7-alpine

ADD . /code
WORKDIR /code

ENV TZ=Europe/Moscow

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# ENV export FLASK_ENV=development

ENV FLASK_DEBUG 0
# ENV FLASK_APP app.py

# CMD ["python3", "app.py"]
CMD ["flask", "run", "--host=0.0.0.0"]
# ENTRYPOINT ["python", "-m", "flask", "run"] # , "--host=0.0.0.0"
