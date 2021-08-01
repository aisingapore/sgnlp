FROM python:3.8-buster

COPY . /nea

WORKDIR /nea

RUN pip install -r requirements.txt

RUN python -m download_pretrained

CMD gunicorn --bind 0.0.0.0:8000 wsgi:app --timeout 180
