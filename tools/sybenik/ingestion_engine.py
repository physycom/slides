#! /usr/bin/env python3

import os
import json
from datetime import datetime
from dateutil import tz
import mysql.connector
import shutil, os
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

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

    logHandler = TimedRotatingFileHandler(logfile, when='midnight', backupCount=14)
    logFormatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S%z")
    logHandler.setFormatter(logFormatter)
    logger = logging.getLogger('ing-logger')
    logger.addHandler(logHandler)
    logger.setLevel(logging.DEBUG)
    self.logger = logger

    self.config = config

    self.HERE = tz.tzlocal()
    self.UTC = tz.gettz('UTC')

    try:
      config = self.config['db']['local']
      db_name = config['database']
      db = mysql.connector.connect(
        host=config['host'],
        port=config['port'],
        database=db_name,
        user=config['user'],
        passwd=config['password']
      )

      query = f"SELECT CAM_NAME, BARRIER_NAME, DIRECTION, UID FROM {db_name}.barriers_meta"
      df = pd.read_sql(query, con=db)
      #print(df)
      self.bar_proxy = {}
      for cn, bn, di, uid in df.values:
        self.bar_proxy[(cn, bn, di)] = uid

      query = f"SELECT CAM_NAME, UID FROM {db_name}.cam_meta"
      df = pd.read_sql(query, con=db)
      #print(df)
      self.cam_proxy = {}
      for cn, uid in df.values:
        self.cam_proxy[cn] = uid

    except Exception as e:
      raise Exception(f'Error in query for cam/barriers : {e}')

    self.logger.info('init engine')


  def do_task(self):
    tnow = datetime.now()
    ts = tnow.timestamp()

    if int(ts - 5) % self.clock_dt == 0:
      if self.pending_ingestion:
        self.logger.info('Ingesting')
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
          dbs = self.config['db']
          for dbtag, dbdata in dbs.items():
            #self.logger.info(f'Pushing to db {dbtag}')
            self.push_db(dbdata, tag=dbtag, cdata=cdata, bdata=bdata)

          for f in datafilenames:
            shutil.move(f, self.bckdir)
            #shutil.move(f, os.path.join(self.bckdir, f))

        else:
          self.logger.info('Nothing to ingest')
        self.pending_ingestion = False
    else:
      if not self.pending_ingestion:
        self.pending_ingestion = True


  def push_db(self, config, tag='db', cdata=None, bdata=None):
    self.logger.info(f'Pushing data to {tag}')

    try:
      db = mysql.connector.connect(
        host=config['host'],
        port=config['port'],
        database=config['database'],
        user=config['user'],
        passwd=config['password']
      )
      cursor = db.cursor()
    except Exception as e:
      self.logger.error(f'Error connecting to {tag} : {e}')
      return

    if cdata:
      try:
        query = f"""INSERT IGNORE INTO {config['database']}.cam_cnt (TIMESTAMP, DATETIME, CAM_UID, MEAN, MAX, MIN) VALUES (%s, %s, %s, %s, %s, %s)"""
        cursor.executemany(query, cdata)
        self.logger.info(f'Inserted : {len(cdata)} cam data')
      except Exception as e:
        self.logger.error(f'Cam push failed : {e}')

    if bdata:
      try:
        query = f"""INSERT IGNORE INTO {config['database']}.barriers_cnt (TIMESTAMP, DATETIME, BARRIER_UID, COUNTER) VALUES (%s, %s, %s, %s)"""
        cursor.executemany(query, bdata)
        self.logger.info(f'Inserted : {len(bdata)} bar data')
      except Exception as e:
        self.logger.error(f'Bar push failed : {e}')

    try:
      db.commit()
      cursor.close()
      db.close()
    except Exception as e:
      self.logger.error(f'Error in closing db connection : {e}')


  def ingest_barfile(self, filename):
    datas = []
    with open(filename) as fin:
      #print(filename)
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
      self.logger.info(f'Datafile {filename} imported')
    #print(datas)
    return datas

  def ingest_camfile(self, filename):
    datas = []
    with open(filename) as fin:
      #print(filename)
      alldata = json.load(fin)
      for data in alldata:
        cname = data['cam_name']
        uid = self.cam_proxy[cname]
        ts = data['timestamp']
        timeutc = datetime.fromtimestamp(ts).replace(tzinfo=self.HERE).astimezone(self.UTC).strftime('%Y-%m-%d %H:%M:%S')
        cntmean = data['counter']['MEAN']
        cntmax = data['counter']['MAX']
        cntmin = data['counter']['MIN']
        datas.append([ ts, timeutc, uid, cntmean, cntmax, cntmin ])
      self.logger.info(f'Datafile {filename} imported')
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
