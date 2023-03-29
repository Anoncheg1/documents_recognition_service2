FROM python:3.7-slim

ADD requirements.txt .

ENV TZ=Europe/Moscow

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
  && pip3 install --upgrade pip \
  && pip3 install --no-cache-dir -r requirements.txt --ignore-installed

ADD . /code
WORKDIR /code

CMD sleep 5 && python3 worker.py
