FROM python:3.8-buster

COPY ./demo_api /demo_api
COPY ./sgnlp /sgnlp
COPY ./setup.py /setup.py
COPY ./README.md /README.md

RUN pip install -r /demo_api/ufd/requirements_dev.txt

WORKDIR /demo_api/ufd
RUN mkdir -p models
RUN python -m download_pretrained

CMD PYTHONPATH=../../ gunicorn -c ../gunicorn.conf.py
