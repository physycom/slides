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
After the building process it will be created a bin directory with the compiled program:
To run the program it is necessary to create a configuration file that will contain all the input variables needed to run. 
#### Configuration File
An example of configuration file (config_file.json):
```
{
  "file_pro": "/path/to/cartography/name_cartography.pro",
  "file_pnt": "/path/to/cartography/name_cartography.pnt",
  "enable_population": true,
  "enable_netstate": true,
  "dt": 10,
  "sampling_dt": 60,
  "enable_gui": true,
  "sampling_graphics": 20,
  "start_date": "2021-07-14 23:00:00",
  "stop_date": "2021-07-15 23:59:59",
  "state_basename": "/path/to/output/name_output_file",
  "file_barrier":"/path/to/directory_barrier_config/barriers_config.csv",
  "attractions": {
    "attraction1": {
      "lat": 45.4376935,
      "lon": 12.3303509,
      "weight": [
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6
      ],
      "timecap": [
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0
      ],
      "visit_time": 2880.0
    },
    "attraction2": {
      "lat": 45.4322447,
      "lon": 12.3367426,
      "weight": [
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6
      ],
      "timecap": [
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0
      ],
      "visit_time": 2880.0
    },
  "sources": {
    "source1": {
      "creation_dt": 30,
      "creation_rate": [
        19.5,
        19.5,
        19.5,
        19.5,
        22.75,
        22.75,
        22.75,
        23.75,
        7.5,
        7.5,
        7.5,
        7.5,
        16.25,
        16.25,
        16.25,
        17.25,
        35.25,
        35.25,
        35.25,
        36.25,
        110.25,
        110.25,
        110.25,
        111.25,
        287.0,
        287.0,
        287.0,
        287.0,
        323.0,
        323.0,
        323.0,
        323.0,
        317.5,
        317.5,
        317.5,
        317.5,
        310.0,
        310.0,
        310.0,
        310.0,
        323.5,
        323.5,
        323.5,
        323.5,
        288.5,
        288.5,
        288.5,
        288.5,
        290.75,
        290.75,
        290.75,
        291.75,
        304.25,
        304.25,
        304.25,
        305.25,
        332.5,
        332.5,
        332.5,
        332.5,
        372.25,
        372.25,
        372.25,
        373.25,
        437.0,
        437.0,
        437.0,
        437.0,
        387.25,
        387.25,
        387.25,
        388.25,
        311.5,
        311.5,
        311.5,
        311.5,
        202.25,
        202.25,
        202.25,
        203.25,
        163.25,
        163.25,
        163.25,
        164.25,
        173.5,
        173.5,
        173.5,
        173.5,
        177.5,
        177.5,
        177.5,
        177.5,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "source_location": {
        "lat": 45.43789,
        "lon": 12.32116
      },
      "pawns_from_weight": {
        "tourist": {
          "beta_bp_miss": 0,
          "speed_mps": 1.0
        }
      }
    },
    "source2": {
      "creation_dt": 30,
      "creation_rate": [
        25.31643669035061,
        22.41276923920396,
        21.893471809303282,
        17.410329687345556,
        14.53311349871294,
        15.358062758460822,
        13.729693811888593,
        12.554863550098533,
        10.949715328801727,
        15.397054488664365,
        13.341145032641224,
        15.612990439721951,
        16.534158033315656,
        15.662058766336997,
        20.220367319530435,
        19.712015739979137,
        27.345610578846074,
        33.07911366835633,
        30.97833716388626,
        39.386718367121816,
        34.29485304221983,
        46.65348060123216,
        53.375347168018884,
        67.27777795798569,
        76.09967712021219,
        86.61360290731697,
        93.91383032470786,
        148.64796431238878,
        130.60987127203884,
        128.86170605000638,
        173.3464528194397,
        146.13150745201554,
        156.19966223950172,
        160.13579350634458,
        156.920597828846,
        178.9470315747615,
        139.42459852173602,
        145.67890098478554,
        140.2935624424499,
        139.5880074886128,
        122.61862657335733,
        108.893624916552,
        125.27637121752221,
        100.08423011833635,
        102.46286040639568,
        89.71704440555972,
        86.60945173583008,
        90.90859933052108,
        93.96727949507962,
        78.6759120459176,
        91.9207690004127,
        84.94257258091943,
        86.98641559257577,
        109.7592976264644,
        97.86785733476503,
        97.00282664262997,
        110.67644480689579,
        110.71434719342636,
        105.57764457819772,
        94.79283376781132,
        115.44162907163471,
        95.47000101065895,
        106.75041597864323,
        128.21327893267255,
        123.51299057665543,
        112.18808019446533,
        118.41066128015305,
        110.6898835966759,
        133.41746185045218,
        115.35954169693562,
        105.18963881114014,
        100.42878994628195,
        95.49697737149197,
        89.1295087562522,
        73.22652565482892,
        99.98979682102208,
        86.99285894556759,
        82.56054045212504,
        71.62467008160208,
        75.79087646617673,
        61.099808189354036,
        58.078193279191986,
        47.633174364771456,
        44.647229577449146,
        56.8872578914843,
        48.442054837852446,
        56.38804993534728,
        44.4084359129852,
        41.75954689217628,
        34.52443810234645,
        29.6276861336736,
        25.842983491806063,
        37.469093735538166,
        31.244184102305745,
        23.875002915647933,
        24.291472186277907
      ],
      "source_location": {
        "lat": 45.4336926,
        "lon": 12.3420174
      },
      "pawns_from_weight": {
        "tourist": {
          "beta_bp_miss": 0,
          "speed_mps": 1.0
        }
      }
    },
    "LOCALS": {
      "source_type": "ctrl",
      "creation_dt": 30,
      "creation_rate": [
        100,
        700,
        80,
        940,
        107,
        100,
        130,
        1480,
        1480,
        1480,
        1480,
        1480,
        1570,
        1660,
        1750,
        1840,
        1930,
        2020,
        2110,
        2199,
        2200,
        2200,
        2200,
        2199,
        2299,
        2399,
        2499,
        2599,
        2500,
        2799,
        2899,
        2900,
        2900,
        2900,
        2999,
        2900,
        2900,
        2900,
        2900,
        2900,
        3550,
        3899,
        3349,
        3499,
        3549,
        3799,
        3850,
        4000,
        4000,
        4000,
        4000,
        4000,
        4000,
        4000,
        4000,
        4000,
        3900,
        3899,
        3799,
        3600,
        3500,
        3499,
        3700,
        3800,
        3800,
        3799,
        3799,
        3799,
        3899,
        300,
        399,
        299,
        299,
        299,
        699,
        299,
        4699,
        2799,
        2899,
        999,
        200,
        200,
        900,
        800,
        100,
        800,
        100,
        100,
        399,
        600,
        800,
        800,
        400,
        400,
        400,
        400
      ],
      "pawns": {
        "locals": {
          "beta_bp_miss": 0,
          "start_node_lid": -1,
          "dest": -1
        }
      }
    }
  }
}
```
### Explanation variables of config_file
Example `file_pro`:
##### Point file `.pro`

Sample
```
# poly_cid  nodeF_cid nodeT_cid length(m) ? type ? ? max_speed(km/h) oneway name
    202462     346486    346507     29.6  0    5 0 0              30      0 Calle_de_la_Congregazione
    202463     346508    346509     75.5  0    5 0 0              30      0 _
    202464     346502    346511     18.9  0    5 0 0              30      0 Calle_Oslavia
```
Fields
- **poly_cid**        : polyline cartography id number;
- **nodeF_cid**       : FRONT node cartography id number;
- **nodeT_cid**       : TAIL node cartography id number;
- **length(m)**       : road length in meters;
- **?**               : ;
- **type**            : parameter classifing the road type, according to

  | enum macro name | value | description |
  |-----------------|------:|-------------|
  | `TYPE_` |  |  |
  | `TYPE_` |  |  |
  | `TYPE_` |  |  |
  | `TYPE_` |  |  |

- **?**               : ;
- **?**               : ;
- **max_speed(km/h)** : max allowed speed in km/h;
- **oneway**          : specifies if a street is one way or not, according to

  | enum macro name | value | description |
  |-----------------|------:|-------------|
  | `ONEWAY_BOTH`   |     0 | Both ways avalaible |
  | `ONEWAY_TF`     |     1 | One way front to tail |
  | `ONEWAY_FT`     |     2 | One way tail to front |
  | `ONEWAY_CLOSED` |     3 | Road closed |

- **name**            : if present, string containing street name with ` ` (space character) replaced with `_`.

##### Point file `.pnt`

Sample
```
102875 2 4
45449310 12306570
45448470 12308070
45447840 12309160
45442880 12315240
109459 3 3
45441320 12314970
45441350 12315070
45441360 12315180
202392 4 2
45434830 12359030
45434430 12358610
202452 5 3
45427910 12361880
45427760 12361850
45427640 12361800
```
##### References

Detailed description of OSM XML format

- [map features](https://wiki.openstreetmap.org/wiki/Map_Features)

##### Cartography generation pipeline

1. Download the relevant geodata from OSM in XML format. Start from [osm](https://www.openstreetmap.org/) or one of their mirror available at [Planet OSM](https://wiki.openstreetmap.org/wiki/Planet.osm).
2. For graph generation are available other tools that are not ready for publication. Write in private to:
3. 
##### enable population
If true, enables the production of /state_basename/population.csv with the following columns:
`datetime;timestamp;locals;tourist;transport;awaiting_transport`

##### enable netstate
If true, enables the production of /state_basename/netstate.csv with the following columns:
`timestamp; id_link_1; ... ; id_link_n`

### Run
From the terminal:
```
cd bin
simengine /path/to/configuration_file/config_file.json
```
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
