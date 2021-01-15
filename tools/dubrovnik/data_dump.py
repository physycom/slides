#! /usr/bin/env python3

import os
import json
import requests
import pandas as pd
#import geopandas as gpd
import matplotlib.pyplot as plt
#import contextily as ctx
import re

class meraki_dumper():

  def __init__(self, config):
    self.config = config
    token = config['apikey']
    self.perPage = 1000
    self.payload  = {}
    self.headers = {
      'Accept': '*/*',
      "X-Cisco-Meraki-API-Key": token
    }
    self.devdatafile = 'station_metadata.csv'

  def create_devices_data(self, devdatafile):
    payload = self.payload
    headers = self.headers
    try:
      url = 'https://api.meraki.com/api/v0/organizations'
      response = requests.request("GET", url, headers=headers, data = payload).json()
      orgid = response[0]['id']
      print('Request for org_id : {orgid}'.format(orgid=orgid))

      url = 'https://api.meraki.com/api/v0/organizations/{orgid}/inventory'.format(orgid=orgid)
      response = requests.request("GET", url, headers=headers, data = payload).json()
      inventory = pd.DataFrame.from_dict(response)
      #print(inventory.columns)
      #print(inventory)
      #print(inventory[['name', 'networkId', 'serial']])

      url = 'https://api.meraki.com/api/v0/organizations/{orgid}/deviceStatuses'
      response = requests.request("GET", url, headers=headers, data = payload).json()
      devices = pd.DataFrame.from_dict(response).sort_values(by=['networkId'])
      #print(devices.columns)
      #print(devices)
      #print(devices[['name', 'serial', 'status']])
      devices.to_csv('devices_status.csv', index=False, header=True, sep=';')

      geodevices=pd.DataFrame()
      for nid, df in inventory.groupby('networkId'):
        print(f'Network ID {nid}, devices {len(df)}')
        url=f'https://api.meraki.com/api/v0/networks/{nid}/devices'
        response = requests.request("GET", url, headers=headers, data = payload).json()
        geodev = pd.DataFrame.from_dict(response)
        geodevices = geodevices.append(geodev)
      #print(geodevices.columns)
      #print(geodevices)
      geodevices.to_csv('devices_meta.csv', index=False, header=True, sep=';')

      devdata = devices[['serial', 'networkId', 'status']].merge(geodevices[['lat', 'lng', 'serial', 'name']], how='outer', left_on='serial', right_on='serial')
      devdata.index.name = 'id'
      devdata.to_csv(devdatafile, sep=';', index=True)
      #print(devdata)
    except Exception as e:
      print(f'Problems with devices metadata : {e}')


  def dump_data(self):
    devdatafile = self.devdatafile
    config = self.config
    payload = self.payload
    headers = self.headers

    start = config['start']
    stop  = config['stop']
    timetag = f"{start.replace(':', '')}_{stop.replace(':', '')}"
    perPage = self.perPage

    if not os.path.exists(devdatafile):
      self.create_devices_data(devdatafile)
    devdata = pd.read_csv(devdatafile, sep=';')

    data = pd.DataFrame()
    product_types = [
      'wireless',
      #'appliance',
      #'switch',
      #'systemsManager',
      #'camera',
      #'cellularGateway',
    ]

    evtdata2 = pd.DataFrame()
    for ptype in product_types:
      evtdata = pd.DataFrame()
      for netid in devdata.networkId.unique():
        tlast = start
        url=f'https://api.meraki.com/api/v0/networks/{netid}/events?productType={ptype}&perPage={perPage}&startingAfter={start}'
        rcnt = 0
        while tlast < stop:
          print(f'Requesting network {netid} prodtype {ptype} #{rcnt:02d}')
          response = requests.request("GET", url, headers=headers, data = payload)
          rcnt += 1
          if response.status_code == 200:
            res = response.json()
            df = pd.DataFrame.from_dict(res['events'])
            chunk_start = res['pageStartAt']
            chunk_stop = res['pageEndAt']
            chunk_len = len(df)
            links = response.headers['Link']
            result = re.search('>; rel=prev, <(.*)>; rel=next, <', links)
            nextlink = result.group(1)

            url = nextlink
            tlast = chunk_stop
            #print(chunk_start, chunk_stop, chunk_len, url)
            print(f'Response : chunk_start {chunk_start}, chunk_len {chunk_len}')

            evtdata = evtdata.append(df)
            evtdata2 = evtdata2.append(df)
          else:
            print(f'Failed request')
            print(response.text)
            break

      evtdata = evtdata.drop(columns=['eventData'])
      evtdata = evtdata.drop_duplicates()
      evtdata = evtdata[ evtdata.occurredAt < stop ]
      evtdata = evtdata.sort_values(by='occurredAt')
      evtdata.to_csv(f'events_{timetag}_{ptype}.csv', sep=';', index=False)
      print(f'***** Eventdata per type {ptype} : {len(evtdata)}')

    evtdata2 = evtdata2.drop(columns=['eventData'])
    evtdata2 = evtdata2.drop_duplicates()
    evtdata2 = evtdata2[ evtdata2.occurredAt < stop ]
    evtdata2 = evtdata2.sort_values(by='occurredAt')
    evtdata2.to_csv(f'events_{timetag}.csv', sep=';', index=False)
    print(evtdata2.columns)
    print(evtdata2)

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("-c", "--conf", help="json config", required=True)
  args = parser.parse_args()

  with open(args.conf) as cin:
    config = json.load(cin)

  dumper = meraki_dumper(config)
  dumper.dump_data()
