FROM jupyter/datascience-notebook:latest

USER root
RUN apt-get update && \
    apt install -y python3-tk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt

USER ${NB_UID}

RUN jupyter labextension install @axlair/jupyterlab_vim && \
    jupyter labextension install @jupyterlab/toc
