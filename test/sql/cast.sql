SELECT ARRAY[[[1,2,3]]]::tensor;
SELECT ARRAY[[1.0, 2.0], [3.0, 4.0]]::tensor;
SELECT ARRAY[1.0,2.0,3.0]::float8[]::tensor;
SELECT ARRAY[[1.0, 2.0],[3.0,4.0]]::float4[]::tensor;
SELECT ARRAY[[1.0, 2.0],[3.0,4.0]]::float8[]::tensor;
SELECT ARRAY[[1.0, 2.0],[3.0,4.0]]::numeric[]::tensor;

SELECT '[[1,2],[3,4]]'::tensor::double precision[];

SELECT '{NULL}'::double precision[]::tensor;
SELECT '{NaN}'::double precision[]::tensor;
SELECT '{Infinity}'::double precision[]::tensor;
SELECT '{-Infinity}'::double precision[]::tensor;
SELECT '{{},{}}'::double precision[]::tensor;
SELECT '{{1, 2},{3,4}}'::double precision[]::tensor;
