# build with
# docker build --tag slides . -f slides.docker

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN apt update && apt upgrade -y
RUN apt install cmake libboost-all-dev -y

RUN python3 -m pip install jsonschema numpy pandas matplotlib scipy
RUN python3 -m pip install requests pykml pymongo mysql-connector-python-rf
RUN python3 -m pip install python-multipart passlib[bcrypt] python-jose[cryptography]

RUN mkdir /root/Code
ENV WORKSPACE /root/Code

RUN mkdir -p -m 700 /root/.ssh
COPY --chown=root:root .ssh/slides_rsa /root/.ssh/id_rsa
COPY --chown=root:root .ssh/slides_rsa.pub /root/.ssh/id_rsa.pub
RUN chmod 600 /root/.ssh/*
RUN ssh-keyscan -H github.com >> /root/.ssh/known_hosts

WORKDIR /root/Code
RUN git config --global user.name "slidesmap" && git config --global user.email slidesmap@gmail.com
RUN git clone git@github.com:physycom/sysconfig.git && git clone git@github.com:pybind/pybind11.git

ENV SLIDES_VERSION "v3.1.2"
RUN \
  echo ${SLIDES_VERSION} > /slides_version && \
  git clone git@github.com:physycom/slides.git && \
  cd slides && \
  git submodule update --init --recursive && \
  ./build.sh

RUN \
  cd /root/Code/slides/ && \
  cp pvt/conf/conf.json.docker vars/conf/conf.json && \
  cp /root/Code/slides/python/sim-ws-oauth2.py /app/main.py

COPY ./prestart.sh /app/prestart.sh
