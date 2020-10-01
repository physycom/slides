# launch with
# uvicorn simulation-ws:app --reload --port 9999

# and test with
# GET test
# curl --request GET http://localhost:9999
# POST test
# curl --header "Content-Type: application/json" --request POST --data '{"key": "value"}' http://localhost:9999/...

import sys
import os
import json
from jsonschema import validate, exceptions
from datetime import datetime
import pandas as pd
from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel, Field
import logging

#################
### libraries ###
#################
major=1
minor=0
tweak=0
ver=f'{major}.{minor}.{tweak}'

#################
### libraries ###
#################
try:
  sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'bin'))
  from pysim import simulation

  sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'python'))
  from conf import conf

except Exception as e:
  raise Exception('library loading error : {}'.format(e)) from e

try:
  schema_file = os.path.join(os.environ['WORKSPACE'], 'slides', 'schema', 'schema_request.json')
  with open(schema_file) as sin:
    schema = json.load(sin)

except Exception as e:
  raise Exception('schema file loading error : {}'.format(e)) from e

# init
tags_metadata = [
  {
    "name" : "version",
    "description": "Welcome GET API reporting version."
  },
  {
      "name": "sim",
      "description": "Simulation API.",
  },
  {
      "name": "poly",
      "description": "Geojson API.",
      "externalDocs": {
          "description": "Geojson specs",
          "url": "https://geojson.org/",
      },
  },
]

app = FastAPI(
  title="SLIDES pedestrian fluxes simulator",
  description="A web api providing pedestrians mobility prediction algorithm",
  version=ver,
  openapi_tags=tags_metadata
)
logger = logging.getLogger("gunicorn.error")

class response_welcome(BaseModel):
  message : str = 'slides simulation ws'
  version : str = ver

class response_sim(BaseModel):
  message : str = 'simulation OK'
  city    : str = 'city name'
  data    : dict = {}

class body_sim(BaseModel):
  start_date : str
  stop_date : str
  sampling_dt : int

class response_poly(BaseModel):
  message : str = 'geojson OK'
  geojson : dict = {}


##########################
#### log function ########
##########################
def logs(s):
  return '{} [sim-ws] {}'.format(datetime.now().strftime('%y%m%d %H:%M:%S'), s)

def log_print(s):
  logger.info(logs(s))

#################
#### APIs ####
#################

@app.get('/',
  response_model=response_welcome,
  tags=['version']
)
async def root():
  return response_welcome()

@app.post(
  '/sim',
  response_model=response_sim,
  tags=['sim']
)
async def sim_post(body: body_sim, request: Request, citytag: str = 'null'):
  client_ip = request.client.host
  log_print('Request from {} city {}'.format(client_ip, citytag))

  start_date = body.start_date
  stop_date = body.stop_date
  sampling_dt = body.sampling_dt
  log_print('Parameters {} - {} sampling {} city {}'.format(start_date, stop_date, sampling_dt, citytag))

  # init conf
  try:
    cfg_file = os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'conf', 'conf.json')
    with open(cfg_file) as cin: cfg = json.load(cin)
    cw = conf(cfg, logger)
  except Exception as e:
    log_print('conf init failed : {}'.format(e))
    raise HTTPException(status_code=500, detail='conf init failed : {}'.format(e))

  # sanity check
  if citytag not in cw.cparams:
    raise HTTPException(status_code=400, detail='city \'{}\' not available. Current cities : {}'.format(citytag, list(cw.cparams.keys())))

  # generate config json
  try:
    simconf = cw.generate(start_date, stop_date, citytag)
  except Exception as e:
    log_print('config generation failed : {}'.format(e))
    raise HTTPException(status_code=500, detail='conf generation failed : {}'.format(e))

  # set up environment and move execution to working dir
  wdir = cw.wdir
  try:
    os.chdir(wdir)
  except:
    os.mkdir(wdir)
    os.chdir(wdir)

  # override sim parameters
  simconf['start_date'] = start_date
  simconf['stop_date'] = stop_date
  simconf['sampling_dt'] = sampling_dt
  simconf['state_basename'] = wdir + '/r_{}'.format(citytag)
  #simconf['explore_node'] = [0]
  confs = json.dumps(simconf)
  with open(wdir + '/wsconf_sim_{}.json'.format(citytag), 'w') as outc: json.dump(simconf, outc, indent=2)
  #print(confs, flush=True)

  # run simulation
  s = simulation(confs)
  log_print('sim info : {}'.format(s.sim_info()))
  if s.is_valid():
    tsim = datetime.now()
    s.run()
    log_print('{} simulation done in {}'.format(citytag, datetime.now() - tsim))

    pof = s.poly_outfile()
    log_print('Polyline counters output file : {}'.format(pof))
    dfp = pd.read_csv(pof, sep = ';')
    pp = { t : { i : int(x) for i, x in enumerate(v[1:]) if x != 0 } for t, v in zip(dfp.timestamp, dfp.values)}

    ret = {
      'message' : 'simulation OK',
      'city' : citytag,
      'data' : pp
    }
  else:
    raise HTTPException(status_code=501, detail='simulation init failed')
  return ret

@app.get('/poly',
  response_model=response_poly,
  tags=['poly']
)
async def poly_get(request: Request, citytag: str = ''):
  client_ip = request.client.host
  log_print('Request from {}'.format(client_ip, citytag))

  # init conf
  try:
    cfg_file = os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'conf', 'conf.json')
    with open(cfg_file) as cin: cfg = json.load(cin)
    cw = conf(cfg)

  except Exception as e:
    log_print('conf generation failed : {}'.format(e))
    raise HTTPException(status_code=500, detail='geojson conf init failed : {}'.format(e))

  if citytag not in cw.cparams:
    raise HTTPException(status_code=401, detail='malformed url citytag {} not in {}'.format(citytag, cw.cparams.keys()))

  wdir = cw.wdir
  try:
    os.chdir(wdir)
  except:
    os.mkdir(wdir)
    os.chdir(wdir)

  with open(cw.cparams[citytag]) as tin:
    simconf = json.load(tin)
  confs = json.dumps(simconf)
  with open(wdir + '/wsconf_geojson_{}.json'.format(citytag), 'w') as outc: json.dump(simconf, outc, indent=2)
  #print(confs, flush=True)

  s = simulation(confs)

  if s.is_valid():
    base =  wdir + '/poly_{}'.format(citytag)
    geojf = base + '.geojson'
    if not os.path.exists(geojf):
      s.dump_poly_geojson(base)
    with open(base + '.geojson') as gin:
      geoj = json.load(gin)
    ret = {
      'message' : 'geojson ok',
      'geojson' : geoj
    }
  else:
    raise HTTPException(status_code=501, detail='geojson creation for \'{}\' failed'.format(citytag))

  return ret
