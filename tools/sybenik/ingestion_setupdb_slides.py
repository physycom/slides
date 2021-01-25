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
  db = mysql.connector.connect(
    host=config['host'],
    port=config['port'],
    database=config['database'],
    user=config['user'],
    passwd=config['password']
  )
  cursor = db.cursor()
  #print(tnow, len(data))

  query = 'SHOW DATABASES'
  cursor.execute(query)
  result = cursor.fetchall()
  print(result)

  query = """
  CREATE TABLE IF NOT EXISTS sybenik.counters (
    `ID` INT NOT NULL AUTO_INCREMENT,
    `TIME` TIMESTAMP NOT NULL,
    `CAM_NAME` VARCHAR(20) NOT NULL,
    `BARRIER_NAME` VARCHAR(20) NOT NULL,
    `DIRECTION` VARCHAR(20) NOT NULL,
    `COUNTER` INT NOT NULL,
    PRIMARY KEY (`ID`),
    UNIQUE KEY `UID` (`TIME`, `CAM_NAME`, `BARRIER_NAME`, `DIRECTION`)
  )
  """
  cursor.execute(query)
  print(result)

  query = """CREATE USER IF NOT EXISTS 'slides'@'localhost' IDENTIFIED BY 'slides2020'"""
  cursor.execute(query)

  query = """GRANT SELECT ON sybenik.* TO 'slides'@'localhost'"""
  cursor.execute(query)
