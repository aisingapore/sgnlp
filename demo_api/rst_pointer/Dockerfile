FROM python:3.8-buster

COPY . /demo_api

WORKDIR /demo_api/rst_pointer

RUN pip install -r requirements.txt
RUN python -m download_pretrained

CMD PYTHONPATH=../../ gunicorn -c ../gunicorn.conf.py