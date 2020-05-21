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
from fastapi import FastAPI, Request, HTTPException
import logging

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

#################
#### fastAPI ####
#################
app = FastAPI()
logger = logging.getLogger("gunicorn.error")

##########################
#### log function ########
##########################
def logs(s):
  head = '{} [sim-ws] '.format(datetime.now().strftime('%y%m%d %H:%M:%S'))
  return head + s

def log_print(s):
  logger.info(logs(s))

@app.get('/')
async def root():
  return {'message': 'slides simulation ws'}

@app.post('/sim')
async def sim_walker_post(body: dict, request: Request, city: str = 'ferrara'):
  client_ip = request.client.host
  log_print('Request from {}'.format(client_ip))

  # check body
  try:
    validate(instance=body, schema=schema)
    start_date = body['start_date']
    stop_date = body['stop_date']
    sampling_dt = body['sampling_dt']
    log_print('Parameters {} - {} sampling {} city {}'.format(start_date, stop_date, sampling_dt, city))
  except Exception as e:
    raise HTTPException(status_code=400, detail='invalid request payload : {}'.format(e))

  # init conf
  try:
    cfg_file = os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'conf.json')
    with open(cfg_file) as cin: cfg = json.load(cin)
    cw = conf(cfg, logger)
  except Exception as e:
    log_print('walker config generation failed : {}'.format(e))
    raise HTTPException(status_code=500, detail='conf init failed : {}'.format(e))

  if city not in cw.cparams:
    raise HTTPException(status_code=400, detail='city \'{}\' not available. Current cities : {}'.format(city, list(cw.cparams.keys())))

  try:
    simconf = cw.generate(start_date, stop_date, city)
  except Exception as e:
    log_print('walker config generation failed : {}'.format(e))
    raise HTTPException(status_code=500, detail='conf generation failed : {}'.format(e))

  wdir = cw.wdir
  try:
    os.chdir(wdir)
  except:
    os.mkdir(wdir)
    os.chdir(wdir)

  simconf['start_date'] = start_date
  simconf['stop_date'] = stop_date
  simconf['sampling_dt'] = sampling_dt
  simconf['state_basename'] = wdir + '/r_{}'.format(city)
  #simconf['explore_node'] = [0]

  confs = json.dumps(simconf)

  with open(wdir + '/wsconf_sim_{}.json'.format(city), 'w') as outc: json.dump(simconf, outc, indent=2)
  #print(confs, flush=True)

  s = simulation(confs)

  if s.is_valid():
    tsim = datetime.now()
    s.run()
    log_print('{} simulation done in {}'.format(city, datetime.now() - tsim))

    pof = s.poly_outfile()
    log_print('Polyline counters output file : {}'.format(pof))
    dfp = pd.read_csv(pof, sep = ';')
    pp = { t : { i : int(x) for i, x in enumerate(v[1:]) if x != 0 } for t, v in zip(dfp.timestamp, dfp.values)}

    ret = {
      'message' : 'simulation OK',
      'city' : city,
      'data' : pp
    }
  else:
    raise HTTPException(status_code=501, detail='simulation init failed')
  return ret

@app.get('/poly')
async def sim_walker_post(request: Request, city: str = 'ferrara'):
  client_ip = request.client.host
  log_print('Request from {}'.format(client_ip))

  # init conf
  try:
    cfg_file = os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'conf.json')
    with open(cfg_file) as cin: cfg = json.load(cin)
    cw = conf(cfg)
  except Exception as e:
    log_print('conf generation failed : {}'.format(e))
    raise HTTPException(status_code=500, detail='geojson conf init failed : {}'.format(e))

  wdir = cw.wdir
  try:
    os.chdir(wdir)
  except:
    os.mkdir(wdir)
    os.chdir(wdir)

  with open(cw.cparams[city]) as tin:
    simconf = json.load(tin)
  confs = json.dumps(simconf)
  with open(wdir + '/wsconf_geojson_{}.json'.format(city), 'w') as outc: json.dump(simconf, outc, indent=2)
  #print(confs, flush=True)

  s = simulation(confs)

  if s.is_valid():
    base =  wdir + '/poly_{}'.format(city)
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
    raise HTTPException(status_code=501, detail='geojson creation for \'{}\' failed'.format(city))

  return ret
