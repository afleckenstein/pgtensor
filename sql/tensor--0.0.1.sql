CREATE TYPE tensor;

CREATE FUNCTION tensor_in(cstring, oid, integer) RETURNS tensor
       AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION tensor_out(tensor) RETURNS cstring
       AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION tensor_recv(internal, oid, integer) RETURNS tensor
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION tensor_send(tensor) RETURNS bytea
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION tensor_add(tensor, tensor) RETURNS tensor
       AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE TYPE tensor (
       INPUT = tensor_in,
       OUTPUT = tensor_out,
       RECEIVE = tensor_recv,
       SEND = tensor_send
);

-- tensor cast functions

CREATE FUNCTION array_to_tensor(integer[], integer, boolean) RETURNS tensor
       AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_tensor(real[], integer, boolean) RETURNS tensor
       AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_tensor(double precision[], integer, boolean) RETURNS tensor
       AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_tensor(numeric[], integer, boolean) RETURNS tensor
       AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION tensor_to_float8(tensor, integer, boolean) RETURNS double precision[]
       AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;


-- tensor casts

CREATE CAST (tensor AS double precision[])
       WITH FUNCTION tensor_to_float8(tensor, integer, boolean) AS IMPLICIT;

CREATE CAST (integer[] AS tensor)
       WITH FUNCTION array_to_tensor(integer[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (real[] AS tensor)
       WITH FUNCTION array_to_tensor(real[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (double precision[] AS tensor)
       WITH FUNCTION array_to_tensor(double precision[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (numeric[] AS tensor)
       WITH FUNCTION array_to_tensor(numeric[], integer, boolean) AS ASSIGNMENT;
