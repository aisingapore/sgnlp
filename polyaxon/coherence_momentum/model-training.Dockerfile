FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

ARG REPO_DIR="."
ARG PROJECT_USER="aisg"
ARG HOME_DIR="/home/$PROJECT_USER"

COPY $REPO_DIR nlp-hub-gcp
WORKDIR $REPO_DIR/nlp-hub-gcp

RUN pip install -r polyaxon/coherence_momentum/requirements.txt
RUN groupadd -g 2222 $PROJECT_USER && useradd -u 2222 -g 2222 -m $PROJECT_USER
RUN chown -R 2222:2222 $HOME_DIR && \
    rm /bin/sh && ln -s /bin/bash /bin/sh
USER 2222

