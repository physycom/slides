#! /usr/bin/env python3

import os
import sys
import json
from datetime import datetime, timedelta
from requests import post, get
import shutil

try:
  sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'utils'))
  from sim_plotter import sim_plot
  from sim_stats import sim_stats
  from sim_wroutes import sim_wroutes
except Exception as e:
  raise Exception('[scan_ws] library load failed : {}'.format(e))

if __name__ == '__main__':
  # parse cli and config
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', required=True)
  parser.add_argument('-m', '--mode', default='sim', choices=['geo', 'sim'])
  parser.add_argument('-s', '--scan', action='store_true')
  args = parser.parse_args()

  with open(args.config) as cin:
    config = json.load(cin)

  mode = args.mode
  base = args.config
  wdir = base[:base.rfind('.')]
  if not os.path.exists(wdir): os.mkdir(wdir)

  url_list = config['url_list']
  cities = config['cities']

  schedule = {}
  sid = 0
  date_format = '%Y-%m-%d %H:%M:%S'
  short_format = '%y%m%d_%H%M%S'

  if 'sim_generation' in config:
    gencfg = config['sim_generation']
    start_date = gencfg['start_date']
    stop_date  = gencfg['stop_date']
    day_dt = gencfg['day_dt']
    sim_dt = [ dtm * 60 for dtm in gencfg['sim_dt_min'] ]
    sim_start_times = gencfg['sim_start_times']

    start = datetime.strptime(start_date, date_format)
    stop = datetime.strptime(stop_date, date_format)

    t = start
    while t < stop:
      for url in url_list:
        for tstart in sim_start_times:
          for sdt in sim_dt:
            for city in cities:
              t0 = datetime.strptime(t.strftime('%Y-%m-%d {}'.format(tstart)), date_format)
              t1 = t0 + timedelta(seconds=sdt)
              schedule.update({
                f'sim_{sid:04d}' : {
                  'start_date'  : t0.strftime(date_format),
                  'stop_date'   : t1.strftime(date_format),
                  'sampling_dt' : 900,
                  'city'        : city,
                  'url'         : url,
                  'out_type'    : 'both',
                }
              })
              sid += 1
      t += timedelta(days=day_dt)

  if 'sim_list' in config:
    for url in url_list:
      for city in cities:
        for start, stop in config['sim_list']:
          schedule.update({
            f'sim_{sid:04d}' : {
              'start_date'  : start,
              'stop_date'   : stop,
              'sampling_dt' : 900,
              'city'        : city,
              'url'         : url,
              'out_type'    : 'both',
            }
          })
          sid += 1

  with open(wdir + '/scan_schedule.json', 'w') as sout:
    json.dump(schedule, sout, indent=2)
  print(f'Scheduled {len(schedule)} requests')

  try:
    with open(os.path.join(os.environ['WORKSPACE'], 'slides', 'vars', 'conf', 'conf.json')) as wscin:
      wsc = json.load(wscin)
    wsdir = wsc['work_dir']
    print(f'Found local ws data in {wsdir}, pngs enabled.')
    do_png = True
    do_png = False
  except Exception as e:
    print(f'Unable to locate ws working dir, skipping pngs, err : {e}')
    do_png = False

  citymap = {} # to avoid repetitions in geojson case
  if args.scan:
    for url in url_list:
      for i, (sid, s) in enumerate(schedule.items()):
        # authentication
        user = config['user']
        pwd = config['pwd']
        res = post(
          f'{url}/login',
          headers={
            "accept" : "application/json",
            "Content-Type" : "application/x-www-form-urlencoded"
          },
          data={
            'username' : user,
            'password' : pwd
          },
          timeout=300
        )
        token = res.json()['access_token']

        # data request
        tsim = datetime.now()
        if mode == 'sim':
          city = s['city']
          cityurl = f'{url}/sim?citytag={city}'
          print(f'Requesting SIM for {city} with params {s}')

          res = post(
            cityurl,
            headers={
              'accept': 'application/json',
              'Content-Type': 'application/json',
              'Authorization': f'Bearer {token}'
            },
            data=json.dumps(s),
            timeout=300
          )
          outname = f'{sid:>04s}_response.json'

          # make plot (only localhost scan)
          try:
            rjson = res.json()
            #print(rjson.keys())
            simid = rjson['sim_id']

            if do_png:
              conf = f'{wsdir}/wsconf_sim_{city}.json'
              conf_clone = f'{wdir}/{sid:>04s}_conf.json'
              shutil.copyfile(conf, conf_clone)
              outpng = f'{wdir}/{sid:>04s}_conf.png'
              sim_plot(confin=conf, outpng=outpng, city=city)

              sd = datetime.strptime(s['start_date'], date_format).strftime(short_format)
              popf = f'{wsdir}/r_{city}_{simid}_population_{sd}.csv'
              popf_clone = f'{wdir}/{sid:>04s}_pop.csv'
              shutil.copyfile(popf, popf_clone)
              outpng = f'{wdir}/{sid:>04s}_pop.png'
              sim_plot(popin=popf, outpng=outpng, city=city)

              statsf = f'{wsdir}/r_{city}_{simid}_pstats_{sd}.csv'
              statsf_clone = f'{wdir}/{sid:>04s}_pstats.csv'
              shutil.copyfile(statsf, statsf_clone)
              outbase = f'{wdir}/{sid:>04s}_pstats'
              sim_stats(statsin=statsf, outbase=outbase, city=city)

              wroutef = f'{wsdir}/r_{city}_{simid}_wrstats_{sd}.csv'
              wroutef_clone = f'{wdir}/{sid:>04s}_wrstats.csv'
              shutil.copyfile(wroutef, wroutef_clone)
              outbase = f'{wdir}/{sid:>04s}_wrstats'
              sim_wroutes(wrin=wroutef, outbase=outbase, city=city)
          except Exception as e:
            print(f'Problems in plot : {e}')
            pass

          if res.status_code != 200:
            print('request error : {}'.format(res.content))
          else:
            rjson = res.json()
            with open(f'{wdir}/{outname}', 'w') as jout:
              json.dump(rjson, jout, indent=2)

        elif mode == 'geo':
          city = s['city']
          if city in citymap: continue # to avoid repetitions in geojson case
          citymap[city] = 'ok'

          # get poly geojson
          cityurl = f'{url}/poly?citytag={city}'
          print(f'Requesting poly GEOJSON data for {city} @{cityurl}')
          res = get(cityurl, timeout=180)
          outname = f'{wdir}/{city}_poly.geojson'
          if res.status_code != 200:
            print('request error : {}'.format(res.content))
          else:
            rjson = res.json()
            rjson = rjson['geojson']
            with open(outname, 'w') as jout:
              json.dump(rjson, jout, indent=2)

          # get grid geojson
          cityurl = f'{url}/grid?citytag={city}'
          print(f'Requesting grid GEOJSON data for {city} @{cityurl}')
          res = get(cityurl, timeout=180)
          outname = f'{wdir}/{city}_grid.geojson'

          if res.status_code != 200:
            print('request error : {}'.format(res.content))
          else:
            rjson = res.json()
            rjson = rjson['geojson']
            with open(outname, 'w') as jout:
              json.dump(rjson, jout, indent=2)
