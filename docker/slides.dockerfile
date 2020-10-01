# build with
# docker build --tag slides . -f slides.docker

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN apt update && apt upgrade -y
RUN apt install cmake libboost-all-dev -y

RUN python3 -m pip install jsonschema numpy pandas
RUN python3 -m pip install requests pykml

RUN mkdir /root/Code
ENV WORKSPACE /root/Code

RUN mkdir -p -m 700 /root/.ssh
COPY --chown=root:root .ssh/ator_rsa /root/.ssh/id_rsa
COPY --chown=root:root .ssh/ator_rsa.pub /root/.ssh/id_rsa.pub
RUN chmod 600 /root/.ssh/*
RUN ssh-keyscan -H github.com >> /root/.ssh/known_hosts

WORKDIR /root/Code
RUN git config --global user.name "vivesimulator" && git config --global user.email vivesimulator@gmail.com
RUN git clone git@github.com:physycom/sysconfig.git && git clone git@github.com:pybind/pybind11.git

RUN git clone git@github.com:physycom/slides.git && cd slides && git submodule update --init --recursive && ./build.sh && cd ..

RUN cp /root/Code/slides/vars/conf/conf.json.docker /root/Code/slides/vars/conf/conf.json

RUN cp /root/Code/slides/python/sim-ws.py /app/main.py
COPY ./prestart.sh /app/prestart.sh
