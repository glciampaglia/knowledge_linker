// computes edge weights as the inverse of the in-degree of the incident vertex
START
    m = node(*) 
MATCH
    n-[r]->m
WITH m, count(r) as cnt
SET 
    m.indegree = cnt
WITH m
MATCH
    n-[r]->m
SET
    r.indegweight = (1.0/m.indegree);
//RETURN r;
    
