#! /usr/bin/env python3

import os
import logging
import json
from datetime import datetime
import mysql.connector

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfile', help='engine config json', required=True)
  args = parser.parse_args()

  with open(args.cfile) as cin:
    config = json.load(cin)

  config = config['db']
  db_name = config['database']
  db = mysql.connector.connect(
    host=config['host'],
    port=config['port'],
    database=config['database'],
    user=config['user'],
    passwd=config['password']
  )
  cursor = db.cursor()
  #print(tnow, len(data))

  query = f'CREATE DATABASE IF NOT EXISTS {db_name} DEFAULT CHARACTER SET utf8'
  cursor.execute(query)

  query = 'SHOW DATABASES'
  cursor.execute(query)
  result = cursor.fetchall()
  print(result)

  query = f"""
  CREATE TABLE IF NOT EXISTS {db_name}.barriers_meta (
    `UID` INT NOT NULL PRIMARY KEY UNIQUE KEY COMMENT 'Unique id',
    `CAM_NAME` VARCHAR(20) NOT NULL COMMENT 'Camera name',
    `BARRIER_NAME` VARCHAR(20) NOT NULL COMMENT 'Barrier Name',
    `DIRECTION` VARCHAR(20) NOT NULL COMMENT 'Direction IN/OUT',
    `TAIL_LAT` FLOAT NOT NULL COMMENT 'Barrier TAIL point Latitude (EPSG:4326)',
    `TAIL_LON` FLOAT NOT NULL COMMENT 'Barrier TAIL point Longitude (EPSG:4326)',
    `TAIL_PX` INT NOT NULL COMMENT 'Barrier TAIL point pixel X coordinate',
    `TAIL_PY` INT NOT NULL COMMENT 'Barrier TAIL point pixel Y coordinate',
    `FRONT_LAT` FLOAT NOT NULL COMMENT 'Barrier FRONT point Latitude (EPSG:4326)',
    `FRONT_LON` FLOAT NOT NULL COMMENT 'Barrier FRONT point Longitude (EPSG:4326)',
    `FRONT_PX` INT NOT NULL COMMENT 'Barrier FRONT point pixel X coordinate',
    `FRONT_PY` INT NOT NULL COMMENT 'Barrier FRONT point pixel Y coordinate'
  )
  """
  cursor.execute(query)

  query = f"""
  CREATE TABLE IF NOT EXISTS {db_name}.cam_meta (
    `UID` INT NOT NULL PRIMARY KEY UNIQUE KEY COMMENT 'Unique id',
    `CAM_NAME` VARCHAR(20) NOT NULL COMMENT 'Camera Name',
    `LAT` FLOAT NOT NULL COMMENT 'Camera Latitude (EPSG:4326)',
    `LON` FLOAT NOT NULL COMMENT 'Camera Longitude (EPSG:4326)'
  )
  """
  cursor.execute(query)

  camdatafile = os.path.join(os.environ['SYBENIK_WORKSPACE'], 'tools', 'sybenik', 'data', 'cam_data.json')
  with open(camdatafile) as cin:
    camdataj = json.load(cin)

  bardata = []
  camdata = []
  buid = 0
  cuid = 0
  for cdata in camdataj:
    cname = cdata['name']
    clat = cdata['coords'][0]
    clon = cdata['coords'][1]
    camdata.append([cuid, cname, clat, clon])
    cuid += 1
    for btag, bar in cdata['barriers'].items():
      tlat, tlon, flat, flon = bar['geocoords']
      tx, ty, fx, fy = bar['pixelcoords']
      bardata.append([buid, cname, btag, 'IN', tlat, tlon, tx, ty, flat, flon, fx, fy])
      buid += 1
      bardata.append([buid, cname, btag, 'OUT', tlat, tlon, tx, ty, flat, flon, fx, fy])
      buid += 1
  print(camdata)
  print(bardata)

  query = f"""INSERT IGNORE INTO {db_name}.barriers_meta (UID, CAM_NAME, BARRIER_NAME, DIRECTION, TAIL_LAT, TAIL_LON, TAIL_PX, TAIL_PY, FRONT_LAT, FRONT_LON, FRONT_PX, FRONT_PY) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
  cursor.executemany(query, bardata)
  query = f"""INSERT IGNORE INTO {db_name}.cam_meta (UID, CAM_NAME, LAT, LON) VALUES (%s, %s, %s, %s)"""
  cursor.executemany(query, camdata)
  db.commit()

  query = f"""
  CREATE TABLE IF NOT EXISTS {db_name}.barriers_cnt (
    `ID` INT NOT NULL AUTO_INCREMENT PRIMARY KEY COMMENT 'Record internal id',
    `TIMESTAMP` INT UNSIGNED NOT NULL COMMENT 'Record unix timestamp',
    `DATETIME` TIMESTAMP COMMENT 'Record datetime UTC (use TIMESTAMP for queries)',
    `BARRIER_UID` INT NOT NULL COMMENT 'Barrier unique id (join with barriers_meta for queries)',
    `COUNTER` INT NOT NULL COMMENT 'Counter value',
    UNIQUE KEY `UID` (`TIMESTAMP`, `BARRIER_UID`)
  )
  """
  cursor.execute(query)

  query = f"""
  CREATE TABLE IF NOT EXISTS {db_name}.cam_cnt (
    `ID` INT NOT NULL AUTO_INCREMENT PRIMARY KEY COMMENT 'Record internal id',
    `TIMESTAMP` INT UNSIGNED NOT NULL COMMENT 'Record unix timestamp',
    `DATETIME` TIMESTAMP NOT NULL COMMENT 'Record datetime UTC (use TIMESTAMP for queries)',
    `CAM_UID` INT NOT NULL COMMENT 'Camera unique id (join with cam_meta for queries)',
    `COUNTER` INT NOT NULL COMMENT 'Counter value',
    UNIQUE KEY `UID` (`TIMESTAMP`, `CAM_UID`)
  )
  """
  cursor.execute(query)

  if db_name == 'sybenik':
    query = """CREATE USER IF NOT EXISTS 'slides'@'localhost' IDENTIFIED BY 'slides2020'"""
    cursor.execute(query)
    query = """GRANT SELECT ON sybenik.counters TO 'slides'@'localhost'"""
    cursor.execute(query)
    query = """FLUSH PRIVILEGES"""
    cursor.execute(query)

