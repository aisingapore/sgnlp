FROM python:3.8-buster

COPY ./demo_api /demo_api
COPY ./sgnlp /sgnlp
COPY ./setup.py /setup.py
COPY ./README.md /README.md

RUN pip install -r /demo_api/sentic_gcn/requirements_dev.txt

WORKDIR /demo_api/sentic_gcn

RUN python -m download_pretrained

CMD PYTHONPATH=../../ gunicorn -c ../gunicorn.conf.py