-- tensor

CREATE TABLE t (val tensor);
INSERT INTO t (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[[1,1,1]]'), (NULL);

CREATE TABLE t2 (val tensor);

\copy t TO 'results/tensor.bin' WITH (FORMAT binary)
\copy t2 FROM 'results/tensor.bin' WITH (FORMAT binary)

SELECT * FROM t2;

DROP TABLE t;
DROP TABLE t2;

