FROM python:3.8-buster

COPY . /rumour_detection_twitter

WORKDIR /rumour_detection_twitter

RUN pip install -r requirements.txt

CMD gunicorn --bind 0.0.0.0:8000 wsgi:app --timeout 180
