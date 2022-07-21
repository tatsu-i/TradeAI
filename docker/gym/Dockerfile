FROM ubuntu:bionic

RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt-get install -y --no-install-recommends \
      apt-transport-https \
      ca-certificates \
      vim \
      tmux \
      curl \
      wget \
      netcat \
      tcpdump \
      locales \
      libpq-dev \
      build-essential \
      python3 \
      python3-setuptools \
      python3-pip \
      python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* && \
    echo en_US.UTF-8 UTF-8 > /etc/locale.gen && \
    mkdir -p /usr/share/locale/ && \
    touch /usr/share/locale/locale.alias && \
    locale-gen && \
    update-locale LANG=en_US.UTF-8 && \
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV MODEL_PATH=/data/model

COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY ./script /script/
WORKDIR /script
RUN python3 setup.py install

COPY work /work
WORKDIR /work

COPY entrypoint.sh /entrypoint.sh
COPY wait-for-it.sh /wait-for-it.sh
ENTRYPOINT ["/entrypoint.sh"]
