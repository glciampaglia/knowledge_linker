START
//   m = node(*) 
    m = node:vertices(value='U http://dbpedia.org/resource/Barack_Obama')
MATCH
    n -[r]-> m
WITH
    count(r) AS cnt
UPDATE
    r.deg-weight = 1.0 / cnt 
