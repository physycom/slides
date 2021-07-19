# launch with
# uvicorn sim-ws-oauth2:app --reload --port 9999

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
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel, validator
import logging
import coloredlogs
import glob

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
  title="SLIDES pedestrian on simwsouaut2 fluxes simulator",
  description="A web api providing pedestrians mobility prediction algorithm",
  version=ver,
  openapi_tags=tags_metadata
)

class response_welcome(BaseModel):
  message : str = 'slides simulation ws'
  version : str = ver

class response_sim(BaseModel):
  message : str = 'simulation OK'
  city    : str = 'city name'
  sim_id  : int = '0'
  poly_cnt : dict
  grid_cnt : dict

response_output_types = ['both', 'poly', 'grid']
class body_sim(BaseModel):
  start_date : str
  stop_date : str
  sampling_dt : int
  out_type : str

  @validator('out_type')
  def out_type_in_list(cls, v):
    if v not in response_output_types:
      raise ValueError(f'Field "out_type" must one of {response_output_types}')
    return v

class response_poly(BaseModel):
  message : str = 'geojson OK'
  geojson : dict = {}

class response_grid(BaseModel):
  message : str = 'geojson OK'
  geojson : dict = {}


#################
#### APIs ####
#################

DEFAULT_GRID_SIZE = 100

@app.on_event("startup")
async def startup_event():
  global logger

  if ver == 'local':
    log_folder = 'logs'
    if not os.path.exists(log_folder): os.mkdir(log_folder)

    # config logger
    console_formatter = coloredlogs.ColoredFormatter('%(asctime)s [%(levelname)s] (%(name)s:%(funcName)s) %(message)s', "%H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    time_formatter = logging.Formatter('%(asctime)s [%(levelname)s] (%(name)s:%(funcName)s) %(message)s', "%y-%m-%d %H:%M:%S")
    # time_handler = TimedRotatingFileHandler(f'{log_folder}/slides_ws.log', when='D', backupCount=7) # m H D : minutes hours days
    time_handler = logging.FileHandler(f'{log_folder}/slides_ws.log')
    time_handler.setFormatter(time_formatter)

    # clear uvicorn logger
    logging.getLogger("uvicorn").handlers.clear()
    access_log = logging.getLogger("uvicorn.access")
    access_log.handlers.clear()
    access_log.addHandler(console_handler)
    access_log.addHandler(time_handler)

    logging.basicConfig(
      level=logging.DEBUG,
      handlers=[
        time_handler,   # log file handler
        console_handler # console stream handler
      ]
    )
  else:
    log_folder = '/output/logs'
    if not os.path.exists(log_folder): os.makedirs(log_folder)

    time_formatter = logging.Formatter('%(asctime)s [%(process)d] [%(levelname)s] (%(name)s) %(message)s', "%y%m%d %H:%M:%S")
    # time_handler = TimedRotatingFileHandler(f'{log_folder}/slides_ws.log', when='D', backupCount=7) # m H D : minutes hours days
    time_handler = logging.FileHandler(f'{log_folder}/slides_ws.log')
    time_handler.setFormatter(time_formatter)

    logging.basicConfig(
      level=logging.DEBUG,
      handlers=[
        time_handler
      ]
    )

  logging.getLogger('matplotlib').setLevel(logging.WARNING)
  logging.getLogger('urllib3').setLevel(logging.WARNING)
  logging.getLogger('shapely').setLevel(logging.WARNING)
  logging.getLogger('passlib').setLevel(logging.WARNING)
  logging.getLogger('multipart').setLevel(logging.WARNING)
  logger = logging.getLogger('slides_ws')
  logger.info(f'Setting logger config for version : {ver}')

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
  logger.info(f'Request from {client_ip} city {citytag}'.format())

  start_date = body.start_date
  stop_date = body.stop_date
  sampling_dt = body.sampling_dt
  out_type = body.out_type
  logger.info(f'Parameters {start_date} - {stop_date} sampling {sampling_dt} city {citytag} out_type {out_type}')

  sim_id = int(datetime.now().strftime('%y%m%d%H%M%S%f'))
  logger.info(f'Simulation id {sim_id} ')

  # init conf
  try:
    cfg_file = os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'conf', 'conf.json')
    with open(cfg_file, encoding="utf8") as cin: cfg = json.load(cin)
    cw = conf(cfg)
  except Exception as e:
    logger.error(f'conf init failed : {e}')
    raise HTTPException(status_code=500, detail='conf init failed : {}'.format(e))

  # sanity check
  if citytag not in cw.cparams:
    raise HTTPException(status_code=400, detail='city \'{}\' not available. Current cities : {}'.format(citytag, list(cw.cparams.keys())))

  # generate config json
  try:
    simconf = cw.generate(start_date, stop_date, citytag)
  except Exception as e:
    logger.error(f'config generation failed : {e}')
    raise HTTPException(status_code=500, detail='conf generation failed : {}'.format(e))

  # set up environment and move execution to working dir
  wdir = cw.wdir
  if not os.path.exists(wdir): os.mkdir(wdir)
  statefiledir = f'{wdir}/statefile'
  if not os.path.exists(statefiledir): os.makedirs(statefiledir)

  try:
    os.chdir(wdir)
  except:
    os.mkdir(wdir)
    os.chdir(wdir)

  # override sim parameters
  simconf['start_date'] = start_date
  simconf['stop_date'] = stop_date
  simconf['sampling_dt'] = sampling_dt
  basename = statefiledir + '/r_{}_{}'.format(citytag, sim_id)
  simconf['state_basename'] = basename
  simconf['enable_stats'] = True

  simconf['enable_netstate'] = True
  simconf['enable_influxgrid'] = True
  simconf['state_grid_cell_m'] = DEFAULT_GRID_SIZE

  if out_type == 'poly':
    simconf['enable_influxgrid'] = False
  elif out_type == 'grid':
    simconf['enable_netstate'] = False

  confs = json.dumps(simconf)
  with open(basename + '_conf.json', 'w') as outc: json.dump(simconf, outc, indent=2)
  #print(confs, flush=True)

  # run simulation
  s = simulation(confs)
  logger.info(f'sim info : {s.sim_info()}')
  if s.is_valid():
    tsim = datetime.now()
    s.run()
    logger.info(f'{citytag} simulation done in {datetime.now() - tsim}')

    if out_type == 'poly' or out_type == 'both':
      pof = s.poly_outfile()
      logger.info(f'Polyline counters output file : {pof}')
      dfp = pd.read_csv(pof, sep = ';')
      poly_cnt = { t : { i : int(x) for i, x in enumerate(v[1:]) if x != 0 } for t, v in zip(dfp.timestamp, dfp.values)}
    else:
      poly_cnt = None

    if out_type == 'grid' or out_type == 'both':
      pof = s.grid_outfile()
      logger.info(f'Grid counters output file : {pof}')

      dfp = pd.read_csv(pof, sep=" |,|=", usecols=[2,4,5], engine='python')
      dfp.columns = ['id', 'cnt', 'timestamp']
      dfp.timestamp = (dfp.timestamp * 1e-9).astype('int')
      #dfp['datetime'] = pd.to_datetime(dfp.timestamp, unit='s', utc=True).dt.tz_convert('Europe/Rome')
      dfp = dfp[ dfp.cnt != 0 ]
      grid_cnt = {
          int(ts) : { str(gid) : int(val) for gid, val in dft[['id', 'cnt']].astype(int).values }
        for ts, dft in dfp.groupby('timestamp')
      }
    else:
      grid_cnt = None

    ret = response_sim(
      message='simulation OK',
      city=citytag,
      sim_id=sim_id,
      poly_cnt=poly_cnt,
      grid_cnt=grid_cnt,
    )

    if cw.remove_local_output:
      print(basename)
      garbage = glob.glob(f'{basename}*')
      for trash in garbage:
        os.remove(trash)
      #print(garbage)

  else:
    raise HTTPException(status_code=501, detail='simulation init failed')
  return ret

@app.get('/poly',
  response_model=response_poly,
  tags=['poly']
)
async def poly_get(request: Request, citytag: str = ''):
  client_ip = request.client.host
  logger.info(f'Request from {client_ip}:{citytag}')

  # init conf
  try:
    cfg_file = os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'conf', 'conf.json')
    with open(cfg_file) as cin: cfg = json.load(cin)
    cw = conf(cfg)

  except Exception as e:
    logger.error(f'conf generation failed : {e}')
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
  logger.info(f'Request from {client_ip} for {citytag} grid')

  # init conf
  try:
    cfg_file = os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'conf', 'conf.json')
    with open(cfg_file) as cin: cfg = json.load(cin)
    cw = conf(cfg)

  except Exception as e:
    logger.error('conf generation failed : {}'.format(e))
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

  simconf['state_grid_cell_m'] = DEFAULT_GRID_SIZE

  confs = json.dumps(simconf)
  with open(wdir + '/wsconf_grid_{}.json'.format(citytag), 'w') as outc: json.dump(simconf, outc, indent=2)
  #print(confs, flush=True)

  s = simulation(confs)

  if s.is_valid():
    base =  wdir + '/grid_{}'.format(citytag)
    geojf = base + '.geojson'
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


if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, port=10002)
