FROM python:3.7-buster

COPY ./demo_api /demo_api
COPY ./sgnlp /sgnlp
COPY ./setup.py /setup.py
COPY ./README.md /README.md

RUN pip install -r /demo_api/lif_3way_ap/requirements_dev.txt

WORKDIR /demo_api/lif_3way_ap

RUN python -m download_pretrained

CMD PYTHONPATH=../../ gunicorn -c ../gunicorn.conf.py