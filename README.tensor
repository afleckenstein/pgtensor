# array computing for postgres

pgtensor adds a tensor datatype to PostgreSQL.

## usage

### basic insertion
```sql
postgres=# CREATE EXTENSION tensor;
postgres=# CREATE TABLE data (t tensor);
postgres=# INSERT INTO data VALUES ('[[1,2],[3,4],[5,6]]');
```

### elementwise add
```sql
postgres=# SELECT tensor_add('[[1,2],[3,4]]', '[[5, 6],[7,8]]');

   tensor_add
-----------------
 [[6,8],[10,12]]
(1 row)
```

## Roadmap (in no particular order):
- Reshape (Transpose)
- Slicing
- To/From Array
- Elementwise Operations
- Linear algebra primitives
- Broadcasting
- OpenBLAS integration
- Tiling (c.f. e.g. PostGIS ST_SUBDIVIDE)
- Expanded object form for efficient in-memory computation