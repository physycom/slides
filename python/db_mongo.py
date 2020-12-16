#! /usr/bin/env python3

import pymongo
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from bson import json_util, ObjectId

from pandas.io.json import json_normalize
import json

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--cfg', help='config file', required=True)
  args = parser.parse_args()

  with open(args.cfg) as f:
    config = json.load(f)

  try:
    client = pymongo.MongoClient(host=          config['host'],
                                 port=          config['port'],
                                 username=      config['user'],
                                 password=      config['pwd'],
                                 authSource=    config['db'],
                                 authMechanism= config['aut'],
                                 )

    ##### TRIPADVISOR ##########
    """
    cursor = client["symfony"].TripAdvisor.aggregate([
      {
        '$project' : {
          'location_id' : 1,
          'city' : 1,
          'name' : 1,
          'review_number' : '$values.num_reviews',
          'category_name' : '$category.name',
          'subcat_name' : '$subcategory.name'
        }
      },
      { '$unwind'  : '$subcat_name'},
      { '$unwind'  : '$review_number'},
      {
        "$group": {
          "_id": "$location_id",
          "name" : { "$first" : "$name" },
          "city": { "$first" : "$city" },
          "review_number": { "$first" : "$review_number" },
          "category_name": { "$first" : "$category_name" },
          "subcat_name"  : { "$first" : "$subcat_name" },
        }
      }
    ])
    """
    cursor = client["symfony"].TripAdvisor.find(
     {
     },
     {
       #'location_id' : 1,
       'city' : 1,
       #'name' : 1,
       'values.num_reviews' : 1
     }
    )
    df = pd.DataFrame(list(cursor))
    print(df)


  except Exception as e:
    print('Connection error : {}'.format(e))
