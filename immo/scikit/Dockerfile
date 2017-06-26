FROM ubuntu:17.04
MAINTAINER Piero Steinger <piero@steinger.eu>, Nicolas Mauchle <nicolas@nicolasmauchle.ch>

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get update \
 && apt-get install -y font-manager python3.6 python3-pip vim python3-tk locales git python3-setuptools \
 && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip3 install sklearn pandas matplotlib seaborn sympy h5py joblib nltk

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    echo 'LANG="en_US.UTF-8"'>/etc/default/locale && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8

# Install all packages for nltk
RUN python3 -m nltk.downloader all

# Xgboost
RUN git clone --recursive https://github.com/dmlc/xgboost \
 && cd xgboost; make -j8 \
 && cd python-package; python3 setup.py install

WORKDIR /usr/src/app/