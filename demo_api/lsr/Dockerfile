FROM python:3.8-buster

COPY . /demo_api

WORKDIR /demo_api/lsr

RUN pip install -r requirements.txt
RUN python -m download_pretrained

CMD PYTHON_PATH=../../ gunicorn -c ../gunicorn.conf.py