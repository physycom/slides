# slides
This repo contains the codebase for SLIDES project.

### Contents

### Launch local webservice container

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
