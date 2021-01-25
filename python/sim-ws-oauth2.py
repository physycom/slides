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
from datetime import datetime
import pandas as pd
from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel, Field
import logging

###############
### version ###
###############
if 'SLIDES_VERSION' in os.environ:
  ver = os.environ['SLIDES_VERSION']
else:
  ver = 'local'

#################
### libraries ###
#################
try:
  sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'bin'))
  from pysim import simulation

  sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'python'))
  from conf import conf
  from security import *
except Exception as e:
  raise Exception('library loading error : {}'.format(e)) from e

# init
tags_metadata = [
  {
    "name" : "welcome",
    "description": "Welcome API."
  },
  {
      "name": "login",
      "description": "OAUTH2 authentication API.",
  },
  {
      "name": "sim",
      "description": "Simulation API.",
  },
  {
      "name": "poly",
      "description": "Geojson cartography API.",
      "externalDocs": {
          "description": "Geojson specs",
          "url": "https://geojson.org/",
      },
  },
  {
      "name": "grid",
      "description": "Geojson grid API.",
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
logger = logging.getLogger("uvicorn")

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

class response_grid(BaseModel):
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
  tags=['welcome']
)
async def root():
  return response_welcome()

@app.post('/login',
  response_model=Token,
  tags=['login']
)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
  user = authenticate_user(users_db, form_data.username, form_data.password)
  if not user:
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail='Incorrect username or password',
      headers={'WWW-Authenticate': 'Bearer'},
    )
  access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
  access_token = create_access_token(
    data={'sub': user.username}, expires_delta=access_token_expires
  )
  return {'access_token': access_token, 'token_type': 'bearer'}

@app.post(
  '/sim',
  response_model=response_sim,
  tags=['sim']
)
async def sim_post(body: body_sim, request: Request, citytag: str = 'null', current_user: User = Depends(get_current_active_user)):
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
  simconf['enable_stats'] = True
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
    raise HTTPException(status_code=500, detail='grid geojson conf init failed : {}'.format(e))

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
  with open(wdir + '/wsconf_poly_{}.json'.format(citytag), 'w') as outc: json.dump(simconf, outc, indent=2)
  #print(confs, flush=True)

  s = simulation(confs)

  if s.is_valid():
    base =  wdir + '/poly_{}'.format(citytag)
    geojf = base + '.geojson'
    if not os.path.exists(geojf):
      s.dump_poly_geojson(base)
    with open(geojf) as gin:
      geoj = json.load(gin)
    ret = {
      'message' : 'geojson ok',
      'type'    : 'poly',
      'geojson' : geoj
    }
  else:
    raise HTTPException(status_code=501, detail='poly geojson creation for \'{}\' failed'.format(citytag))

  return ret

@app.get('/grid',
  response_model=response_grid,
  tags=['grid']
)
async def grid_get(request: Request, citytag: str = ''):
  client_ip = request.client.host
  log_print(f'Request from {client_ip} for {citytag} grid')

  # init conf
  try:
    cfg_file = os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'conf', 'conf.json')
    with open(cfg_file) as cin: cfg = json.load(cin)
    cw = conf(cfg)

  except Exception as e:
    log_print('conf generation failed : {}'.format(e))
    raise HTTPException(status_code=500, detail='grid geojson conf init failed : {}'.format(e))

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
  with open(wdir + '/wsconf_grid_{}.json'.format(citytag), 'w') as outc: json.dump(simconf, outc, indent=2)
  #print(confs, flush=True)

  s = simulation(confs)

  if s.is_valid():
    base =  wdir + '/grid_{}'.format(citytag)
    geojf = base + '.geojson'
    if not os.path.exists(geojf):
      s.dump_grid_geojson(geojf)
    with open(geojf) as gin:
      geoj = json.load(gin)
    ret = {
      'message' : 'geojson ok',
      'type'    : 'grid',
      'geojson' : geoj
    }
  else:
    raise HTTPException(status_code=501, detail='grid geojson creation for \'{}\' failed'.format(citytag))

  return ret
