# slides
This repo contains the codebase for SLIDES project.
### Prerequisite 
The prerequisite for code building are found in :
`https://github.com/physycom/sysconfig/blob/master/setup_eng.md`
### Installation
For the installation from github is sufficient to type in the terminal from the workspace created in the previous step:
`git clone git@github.com:physycom/slides.git`
### Building
From the shell, inside the directory slides run the command:
`./build.sh`

### Usage
#### Configuration File
An example of configuration file (config_file.json):
```
{
  "work_dir" : "/output",
  "remove_local_output": false,

  "kml_data" : {
    "mymaps_id" : {
      "name_city_1"      : "...",
      "name_city_2" : "...",
      "name_city_3"   : "...",
      "name_city_4"   : "...",
      "name_city_5"   : "..."
    }
  },

  "model_data": {

    "model1_file_dir" : [
      "/output/data_real",
      "/output/storico/name_city_1"
    ],

    "params" : {
      "name_city_1" : {
        "population"    : 6000,
        "daily_tourist" : 20000
      },

      "name_city_2":
      {
        "population"    : 10000,
        "daily_tourist" : 5000,

        "mysql" : {
          "host" : "...",
          "port" : ...,
          "user" : "...",
          "pwd"  : "...",
          "db"   : "..."
        },

        "station_mapping" : {
          "Camera_1" : [
            "...",
            "...",
            "..."
          ],
          "Camera_2" : [
            "...",
            "..."
          ],
          "Camera_3": [
            "..."
          ],
          "Camera_4": [
            "...",
            "...",
            "...",
            "...",
            "..."
          ]
        }
      },

      "name_city_3":
      {
        "population"    : 7000,
        "daily_tourist" : 15000,

        "mongo" : {
          "host" : "...",
          "port" : ...,
          "user" : "...",
          "pwd"  : "...",
          "db"   : "...",
          "aut"  : "..."
        },

        "mysql" : {
          "host" : "...",
          "port" : ...,
          "user" : "...",
          "pwd"  : "...",
          "db"   : "..."
        },

        "station_mapping" : {
          "6" : [
            "Stazione"
          ],
          "4" : [
            "parcheggio_1",
            "parcheggio_2"
          ]
        }
      },

      "name_city_4" : {
        "population"    : 1000,
        "daily_tourist" : 20000,

        "mysql" : {
          "host" : "...",
          "port" : ...,
          "user" : "...",
          "pwd"  : "...",
          "db"   : "..."
        }
      },
      "name_city_5" : {
        "population"    : 5000,
        "daily_tourist" : 20000
      }
    }
  },

  "cities" : {
    "name_city_1" : {

    },
    "name_city_2" : {

    },
    "name_city_3" : {

    },
    "name_city_4" : {

    },
    "name_city_5" : {

    }
  },
  
  "start_date" : "{start}",
  "stop_date"  : "{stop}"
}
```
### Explanation variables of config_file
`my_maps_id` : contains city names as keys and identification for the map to  

### Run
After the building process (that should have produced a bin folder in slides) launch with
```uvicorn sim-ws-oauth2:app --reload --port 9999'''
and test with
```
GET test
curl --request GET http://localhost:9999
POST test
curl --header "Content-Type: application/json" --request POST --data '{"key": "value"}' http://localhost:9999/...```
### Further informations
pvt folder contains configurations with sensitive production data and that open access is not available


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
