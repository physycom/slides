# slides
This repo contains the codebase for SLIDES project.

### Contents
+ __Now-casting models__: a model for pedestrian mobility and mobility demand in each destination.
+ __Forecasting models__: visitors’ flows based on mobility demand computed from the experimental observations.

The models are provided as docker containers which expose a series of web APIs whose OpenAPI-compliant technical documentation is available at `http://container_ip:container_port/docs` and `http://container_ip:container_port/redoc`.

### Set up docker container
We choose as build context for docker image the `docker` folder of this repo. A proper docker environment is created a the end of [`build.sh`](build.sh).

A template for docker-compose YAML is provided, make a copy that fit your needs and rename it `docker-compose.yml`.

From within `docker` folder, build the image with
```
> docker-compose build
```
and start daemon container with
```
> docker-compose up -d slides_ws
```
