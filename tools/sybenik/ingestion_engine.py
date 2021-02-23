#! /usr/bin/env python3

import os
import logging
import json
from datetime import datetime
from dateutil import tz
import mysql.connector
import shutil, os
import pandas as pd

class ingestion:


  def __init__(self, configfile):
    with open(configfile) as cin:
      config = json.load(cin)
    self.wdir = config['workdir']
    self.datadir = config['datadir']
    self.bckdir = f'{self.wdir}/backup_data'
    if not os.path.exists(self.bckdir): os.mkdir(self.bckdir)
    logfile = f'{self.wdir}/ingestion-engine.log'
    self.clock_dt = config['clock_dt']
    self.pending_ingestion = True

    logging.basicConfig(
      filename=logfile,
      filemode='w',
      level=logging.DEBUG,
      format='%(asctime)s [%(levelname)s] %(message)s',
      datefmt='%y-%m-%d %H:%M:%S%z'
    )
    self.config = config

    self.HERE = tz.tzlocal()
    self.UTC = tz.gettz('UTC')

    try:
      config = self.config['db']
      self.db_name = config['database']
      db = mysql.connector.connect(
        host=config['host'],
        port=config['port'],
        database=config['database'],
        user=config['user'],
        passwd=config['password']
      )

      query = f"SELECT CAM_NAME, BARRIER_NAME, DIRECTION, UID FROM {self.db_name}.barriers_meta"
      df = pd.read_sql(query, con=db)
      #print(df)
      self.bar_proxy = {}
      for cn, bn, di, uid in df.values:
        self.bar_proxy[(cn, bn, di)] = uid

      query = f"SELECT CAM_NAME, UID FROM {self.db_name}.cam_meta"
      df = pd.read_sql(query, con=db)
      #print(df)
      self.cam_proxy = {}
      for cn, uid in df.values:
        self.cam_proxy[cn] = uid

    except Exception as e:
      raise Exception(f'Error in query for cam/barriers : {e}')

    logging.info('init engine')


  def do_task(self):
    tnow = datetime.now()
    ts = tnow.timestamp()

    if int(ts - 5) % self.clock_dt == 0:
      if self.pending_ingestion:
        logging.info('Ingesting')
        cdata = []
        bdata = []
        datafilenames = []
        for i, f in enumerate(os.listdir(self.datadir)):
          fullf = os.path.join(self.datadir,f)
          if os.stat(fullf).st_mtime < ts - 30:
            if f.startswith('cam_'):
              dataf = self.ingest_camfile(fullf)
              cdata.extend(dataf)
            elif f.startswith('bar_'):
              dataf = self.ingest_barfile(fullf)
              bdata.extend(dataf)
            datafilenames.append(fullf)
          #if i > 1: break

        if len(datafilenames):
          try:
            config = self.config['db']
            db = mysql.connector.connect(
              host=config['host'],
              port=config['port'],
              database=config['database'],
              user=config['user'],
              passwd=config['password']
            )
            cursor = db.cursor()
            query = f"""INSERT IGNORE INTO {self.db_name}.cam_cnt (TIMESTAMP, DATETIME, CAM_UID, COUNTER) VALUES (%s, %s, %s, %s)"""
            cursor.executemany(query, cdata)
            query = f"""INSERT IGNORE INTO {self.db_name}.barriers_cnt (TIMESTAMP, DATETIME, BARRIER_UID, COUNTER) VALUES (%s, %s, %s, %s)"""
            cursor.executemany(query, bdata)
            db.commit()
            cursor.close()
            db.close()
            logging.info(f'Inserted : {len(cdata)} cam data {len(bdata)} cam data ')

            for f in datafilenames:
              shutil.move(f, self.bckdir)

          except Exception as e:
            print(e)
        else:
          logging.info('Nothing to ingest')
        self.pending_ingestion = False
    else:
      if not self.pending_ingestion:
        self.pending_ingestion = True


  def ingest_barfile(self, filename):
    datas = []
    with open(filename) as fin:
      print(filename)
      data = json.load(fin)
      for d in data:
        cname = d['cam_name']
        ts = d['timestamp']
        timeutc = datetime.fromtimestamp(ts).replace(tzinfo=self.HERE).astimezone(self.UTC).strftime('%Y-%m-%d %H:%M:%S')
        #print('oh', ts, timeutc)
        for bname, bdata in d['counter'].items():
          for bdir, dcnt in bdata.items():
            uid = self.bar_proxy[(cname, bname, bdir)]
            datas.append([ ts, timeutc, uid, dcnt ])
      logging.info(f'Datafile {filename} imported')
    #print(datas)
    return datas

  def ingest_camfile(self, filename):
    datas = []
    with open(filename) as fin:
      print(filename)
      data = json.load(fin)
      cname = data['cam_name']
      uid = self.cam_proxy[cname]
      ts = data['timestamp']
      timeutc = datetime.fromtimestamp(ts).replace(tzinfo=self.HERE).astimezone(self.UTC).strftime('%Y-%m-%d %H:%M:%S')
      cnt = data['counter']
      datas.append([ ts, timeutc, uid, cnt ])
      logging.info(f'Datafile {filename} imported')
    #print(datas)
    return datas


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfile', help='engine config json', required=True)
  args = parser.parse_args()

  ing = ingestion(configfile=args.cfile)
  while True:
    ing.do_task()
