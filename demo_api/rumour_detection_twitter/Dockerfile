FROM python:3.8-buster

COPY . /demo_api

WORKDIR /demo_api/rumour_detection_twitter

RUN pip install -r requirements.txt

CMD PYTHONPATH=../../ gunicorn --bind 0.0.0.0:8000 api:app --timeout 180
