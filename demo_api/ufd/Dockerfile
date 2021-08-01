FROM python:3.8-buster

COPY . /ufd
WORKDIR /ufd
RUN mkdir -p models

RUN pip install -r requirements.txt

RUN python -m download_pretrained

CMD gunicorn --bind 0.0.0.0:8000 wsgi:app --timeout 180
