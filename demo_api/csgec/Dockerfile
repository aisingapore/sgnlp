FROM python:3.8-buster

COPY . /demo_api

WORKDIR /demo_api/csgec

RUN pip install -r requirements.txt

CMD PYTHONPATH=../../ gunicorn -c ../gunicorn.conf.py
