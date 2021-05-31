SELECT
  d.id_station,
  d.kind,
  MIN(d.date_time),
  MAX(d.date_time)
FROM 
  Ferrara.DevicesStations d
WHERE 
  #d.kind = 'wifi' AND
  d.date_time > '2021-03-01 00:00:00'
GROUP BY
  d.id_station, d.kind
;
