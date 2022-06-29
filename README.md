# slides
This repo contains the codebase for SLIDES project.
### Usage
#### Configuration File
An example of configuration file `config_file.json`:
```
{
  "work_dir" : "...",
  "remove_local_output": false,

  "kml_data" : {
    "mymaps_id" : {
      "city_name"      : "..."
    }
  },

  "model_data": {
    "params" : {
      "city_name":
      {
        "population"    : ...,
        "daily_tourist" : ...
        }
  },

  "cities" : {
    "city_name" : {

    }
  },
  
  "start_date" : "...",
  "stop_date"  : "..."
}
```
### Explanation variables of config_file.json
`work_dir`: working directory

`remove_local_output`:

`city_name`: name of the city of interest

`population`: number of locals 

`daily_tourist`: number of tourists per day

`start_date`: start date of simulation

`stop_date`: stop date simulation


### Run locally for debugging purposes
After the building process (that should have produced a bin folder in slides) launch with
```uvicorn sim-ws-oauth2:app --reload --port 9999```
and test with
```
GET test
curl --request GET http://localhost:9999
POST test
curl --header "Content-Type: application/json" --request POST --data '{"key": "value"}' http://localhost:9999/...
```
### Further informations
pvt folder contains configurations with sensitive production data whose open access is not available


### Contents
+ __Now-casting models__: a model for pedestrian mobility and mobility demand in each destination.
+ __Forecasting models__: visitors’ flows based on mobility demand computed from the experimental observations.

The models are provided as docker containers which expose a series of web APIs whose OpenAPI-compliant technical documentation is available at `http://container_ip:container_port/docs` and `http://container_ip:container_port/redoc`.

Doc of fastapi for the technical configuration of the container at:
`https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker`



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
