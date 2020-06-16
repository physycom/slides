db = db.getSiblingDB('symfony')
//res = db.GTrends.aggregate([ {"$group" : {_id:"$city", count:{$sum:1}}} ]).toArray()
/*
res = db.TripAdvisor.find(
  {'values.day':'2020-05-16'},
  {
    city : 1,
    'values.subcategory.localized_name' : 1,
  }
).limit(3).toArray()
*/
//res = db.TripAdvisor.explain()
//printjson(res)
q = db.TripAdvisor.find(
  {
    //$unwind : 1
  },
  {
    //name : 1
  }
).limit(1)
res = q.toArray()
//printjson(res)

q = db.TripAdvisor.aggregate([
  {
    $match: { 'location_id': '196476' }
  },
  {
    '$project': {
      'location_id': 1,
      'city': 1,
      'name': 1,
      'review_number': '$values.num_reviews',
      'category_name': '$category.name',
      'subcat_name': '$subcategory.name'
    }
  },
  {
    '$unwind': '$subcat_name'
  },
  {
    '$unwind': '$review_number'
  }
])
res = q.toArray()
//printjson(res)

q = db.TripAdvisor.aggregate([
  {
    '$unwind': '$values'
  },
  {
    '$project': {
      'location_id': 1,
      'city': 1,
      'name': 1,
      //'review_number': '$values.num_reviews',
    }
  },
])
res = q.toArray()
//printjson(res)

q = db.TripAdvisor.aggregate([
  { '$match': { 'location_id': '196476' } },
  { '$unwind': '$values' },
  {
    '$project': {
      'date': '$values.date',
      'city': 1,
      'id': '$location_id',
      'name': '$name',
      'n_rev': '$values.num_reviews',
      'n_top_rev': '$values.review_rating_count.5',
      'group_name': { '$arrayElemAt': ['$groups.name', 0] },
    }
  }
])
res = q.toArray()
//printjson(res)


q = db.TripAdvisor.aggregate([
  { '$match': { 'location_id': '196476' } },
  { '$unwind': '$values' },
  {
    '$project' : {
      'date': '$values.date',
      'city': 1,
      //      'key': 1,
      'id': '$location_id',
      //'category': '$category.name',
      'name': '$name',
      'n_rev': '$values.num_reviews',
      'n_top_rev': '$values.review_rating_count.5',
      'group_name' : { '$arrayElemAt' : [ '$groups.name', 0 ] },
      'triptypes': {
        '$arrayToObject' : {
          '$map': {
            input: "$values.trip_types",
            as: "trip",
            in: [ { '$concat' : [ "triptypes_", "$$trip.name" ] }, "$$trip.value" ]
          }
        }
      }
    }
  }
])
res = q.toArray()
printjson(res)
