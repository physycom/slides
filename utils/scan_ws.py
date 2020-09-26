#! /usr/bin/env python3

import os
import json
import argparse
from datetime import datetime, timedelta
from requests import post, get

if __name__ == '__main__':
  # parse cli and config
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

  start_date = config['start_date']
  stop_date  = config['stop_date']
  day_dt = config['day_dt']
  sim_dt = config['sim_dt']
  sim_start_times = config['sim_start_times']
  url_list = config['url_list']
  cities = config['cities']

  date_format = '%Y-%m-%d %H:%M:%S'
  start = datetime.strptime(start_date, date_format)
  stop = datetime.strptime(stop_date, date_format)

  schedule = {}
  t = start
  sid = 0
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
                'sampling_dt' : 300,
                'city'        : city,
                'url'         : url
              }
            })
            sid += 1
    t += timedelta(days=day_dt)

  with open(wdir + '/scan_schedule.json', 'w') as sout:
    json.dump(schedule, sout, indent=2)
  print(f'Scheduled {len(schedule)} requests')

  citymap = {} # to avoid repetitions in geojson case
  if args.scan:
    for url in url_list:
      for i, (sid, s) in enumerate(schedule.items()):

        tsim = datetime.now()
        if mode == 'sim':
          city = s['city']
          cityurl = f'{url}/sim?citytag={city}'
          print(f'Requesting SIM for {city} with params {s}')
          res = post(cityurl, data=json.dumps(s), timeout=180)
          outname = f'response_{sid:>04s}.json'
        elif mode == 'geo':
          city = s['city']
          if city in citymap:
            continue
          citymap[city] = 'ok'
          cityurl = f'{url}/poly?citytag={city}'
          print(f'Requesting GEOJSON data for {city} @{cityurl}')
          res = get(cityurl, data=json.dumps(s), timeout=180)
          outname = f'geojson_{city}.geojson'

        if res.status_code != 200:
          raise Exception('request error : {}'.format(res.content))
        else:
          rjson = res.json()
          if mode == 'geo':
            rjson = rjson['geojson']
          tsim = datetime.now() - tsim
          with open(f'{wdir}/{outname}', 'w') as jout:
            json.dump(rjson, jout, indent=2)