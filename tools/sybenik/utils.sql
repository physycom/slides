SELECT 
 s.CAM_NAME,
 COUNT(s.TIME),
 MIN(s.TIME),
 MAX(s.TIME) 
FROM 
 counters s
GROUP BY
 s.CAM_NAME
;

SELECT 
 s.*
FROM 
 counters s
WHERE
 s.CAM_NAME = 'Porto' AND
 s.DIRECTION = 'IN'
 
;

SELECT * FROM mysql.user;