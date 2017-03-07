FROM python:3.6-onbuild

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python-dev python-numpy python-setuptools python-scipy libatlas-dev libatlas3-base \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*



WORKDIR /usr/src/app/

