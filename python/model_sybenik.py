#! /usr/bin/env python3

import os
import json
import argparse
import pymongo
import numpy as np
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict

#############################
#### model sybenik class ####
#############################
class model_sybenik():

  def __init__(self, config, logger = None):
    self.config = config
    self.rates_dt = 10 * 60

  def full_table(self, start, stop, tag, resampling=None):
    camera_map ={
      "Porto" : "Port_Of_Šibenik"
    }

    if tag in camera_map.values():

      config = self.config['mysql']

      db = mysql.connector.connect(
        host     = config['host'],
        port     = config['port'],
        user     = config['user'],
        passwd   = config['pwd'],
        database = config['db']
      )
      cursor = db.cursor()

      camera_filter = ' OR '.join([ f"m.CAM_NAME = '{name}'" for name in camera_map ])

      query = f"""
        SELECT
          m.UID,
          m.CAM_NAME
        FROM
          barriers_meta m
        WHERE
          {camera_filter}
      """
      cursor.execute(query)
      result = cursor.fetchall()
      sidconv = { v[0] : v[1] for v in result }
      # print('sid', sidconv)

      query = f"""
        SELECT
          c.DATETIME,
          c.BARRIER_UID,
          c.COUNTER
        FROM
          barriers_cnt c
        WHERE
          c.DATETIME > ('{start}') AND c.DATETIME < ('{stop}')
          AND
          (BARRIER_UID in {tuple(sidconv.keys())} )
      """
      # print('\nquery\n',query)
      tquery = datetime.now()
      cursor.execute(query)
      result = cursor.fetchall()
      # print(result)
      tquery = datetime.now() - tquery
      if len(result) == 0:
        raise Exception(f'[mod_sy] Empty mysql query result')

      df1 = pd.DataFrame(result)
      df1.columns =  cursor.column_names
      df1.index = pd.to_datetime(df1.DATETIME)
      df1 = df1.tz_localize('utc')
      df = df1
      df.index = pd.to_datetime(df.index).tz_localize(None)

      if resampling != None:# and resampling < self.rates_dt:
        resampling_min = resampling // 60

        df = df.groupby(df.index).sum('COUNTER')
        df = df.drop(columns='BARRIER_UID')

        df = df.resample(f'{resampling}s').sum()

        start_date = start.replace(
          minute=resampling_min*(start.minute//resampling_min),
          second=0
        )
        stop_date = stop - timedelta(seconds=1)
        stop_date = stop_date.replace(
          minute=resampling_min*(stop_date.minute//resampling_min),
          second=0
        )

        fullt = pd.date_range(start=start_date,end= stop_date, freq=f'{resampling_min}min')
        df = df.reindex(fullt).interpolate(direction='both')
        df = df.interpolate(limit_direction='both')
        df = df[ (df.index >= start) & (df.index < stop) ]

        df = df.rename(columns = {'COUNTER' : tag})

        data = df

        # def box_centered_kernel(tot_len, box_len):
        #   pad_len = tot_len - box_len
        #   kern = np.concatenate([
        #   np.zeros((pad_len // 2)),
        #   np.ones((box_len)) / box_len,
        #   np.zeros((pad_len - pad_len // 2))# for odd box_len
        #   ])
        #   return kern
        
        # ma_size = 5 # running average idx interval from time in seconds
        # kern = box_centered_kernel(len(data), ma_size)
        # conv = np.fft.fftshift(np.real(np.fft.ifft( np.fft.fft( data.tag ) * np.fft.fft(kern) )))
        # data.tag = conv

        print(data)
        return data

    else:
      raise Exception(f'Station {tag} is virtual')

if __name__ == '__main__':
  pass