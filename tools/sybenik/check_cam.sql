SELECT 
  b.BARRIER_UID,
  bm.CAM_NAME,
  bm.BARRIER_NAME,
  bm.DIRECTION,
  MIN(b.DATETIME),
  MAX(b.DATETIME),
  SUM(b.COUNTER)
FROM
  sibenik.barriers_cnt b
JOIN
  sibenik.barriers_meta bm
ON
  b.BARRIER_UID = bm.UID
GROUP BY
  b.BARRIER_UID
;

SELECT 
  c.CAM_UID,
  cm.CAM_NAME,
  MIN(c.DATETIME),
  MAX(c.DATETIME),
  AVG(c.MEAN),
  AVG(c.MAX),
  AVG(c.MIN)
FROM
  sibenik.cam_cnt c
JOIN
  sibenik.cam_meta cm
ON
  c.CAM_UID = cm.UID
GROUP BY
  c.CAM_UID
;
  