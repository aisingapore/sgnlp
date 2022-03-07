FROM registry.aisingapore.net/polyaxon/cuda10:latest

ARG USER="polyaxon"
ARG WORK_DIR="/home/$USER"

RUN rm /bin/sh && ln -s /bin/bash /bin/sh && \
    apt update && apt install -y jq ca-certificates

WORKDIR $WORK_DIR
USER $USER

COPY build/polyaxon/dual_bert/conda.yml .
RUN conda env update -f conda.yml -n base

WORKDIR /code

COPY --chown=$USER:$USER build .

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV TRANSFORMERS_CACHE="/polyaxon-data/workspace/$USER/.cache"