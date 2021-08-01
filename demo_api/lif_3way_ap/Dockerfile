FROM python:3.8-buster

COPY . /lif_3way_ap/

WORKDIR /lif_3way_ap

RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
RUN python -m download_pretrained

CMD gunicorn --bind 0.0.0.0:8000 wsgi:app --timeout 180