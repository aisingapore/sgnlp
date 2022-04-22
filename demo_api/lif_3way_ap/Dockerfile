FROM python:3.7-buster

COPY . /demo_api

WORKDIR /demo_api/lif_3way_ap

RUN pip install -r requirements.txt
RUN python -m download_pretrained

CMD PYTHONPATH=../../ gunicorn -c ../gunicorn.conf.py