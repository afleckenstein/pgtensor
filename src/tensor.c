#include "postgres.h"

#include <stddef.h>
#include <math.h>

#include "catalog/pg_type.h"
#include "common/shortest_dec.h"
#include "fmgr.h"
#include "libpq/pqformat.h"
#include "nodes/nodeFuncs.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/elog.h"
#include "utils/float.h"
#include "utils/lsyscache.h"

#include "tensor.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#if PG_VERSION_NUM >= 170000
#include "parser/scansup.h"
#endif

PG_MODULE_MAGIC;

typedef enum {
	TTOK_LEVEL_START,
	TTOK_LEVEL_END,
	TTOK_DELIM,
	TTOK_ELEM,
	TTOK_ERROR,
} TensorToken;

static inline void
CheckDims(Tensor * a, Tensor * b)
{
  
	if (a->ndim != b->ndim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different tensor dimensions %d and %d", a->ndim, b->ndim)));
}

static inline void
CheckShapes(Tensor * a, Tensor * b)
{
	CheckDims(a, b);
	for(int i = 0; i < a->ndim; i++) {
		if(a->shape[i] != b->shape[i]) {
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("different tensor shape along dimension %d (%d vs. %d)", i, a->shape[i], b->shape[i])));
		}
	}
}

/*
 * Ensure finite element
 */
static inline void
CheckElement(float value)
{
	if (isnan(value))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("NaN not allowed in tensor")));

	if (isinf(value))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("infinite value not allowed in tensor")));
}


static bool ReadTensorStr(char **srcptr,
						  int32 *ndim_p,
						  int32 *shape,
						  int32 *nitems_p,
						  float8 **values_p,
						  const char *origStr,
						  Node *escontext);

static TensorToken ReadTensorToken(char **srcptr, float8 *val,
								   const char *origStr, Node *escontext);

 
static inline int32 tensor_nitems(Tensor *t) {
	int32 total = 1;
	for (int32 i = 0; i < TENSOR_NDIM(t); i++) {
		total *= TENSOR_SHAPE(t)[i];
	}
	return total;
}

static inline int32 tensor_size(int32 ndim, int32 nitems) {
	int32 shape_offset = offsetof(Tensor, shape);
	int32 shape_size = ndim * sizeof(int32);
	int32 data_size = nitems * sizeof(float8);
	int32 data_start = (shape_offset + shape_size);

	return data_start + data_size + VARHDRSZ;
}


Tensor *InitTensor(int32 ndim, int32 *shape) {
	Tensor *result;
	int32 nelems=1;
	int32 total_size, shape_end;
	for (int32 i = 0; i < ndim; i++) {
		nelems *= shape[i];
	}

	total_size = tensor_size(ndim, nelems);
	result = (Tensor *) palloc0(total_size);
	SET_VARSIZE(result, total_size);
	result->ndim = ndim;
	for(int i = 0; i < ndim; i++) {
		result->shape[i] = shape[i];
	}
	shape_end = offsetof(Tensor, shape) + ndim * sizeof(int);
	result->data_offset = shape_end;

	return result;
}

#if PG_VERSION_NUM >= 170000
#define tensor_isspace(ch) scanner_isspace(ch)
#else
static inline bool
tensor_isspace(char ch) {
	if (ch == ' ' ||
		ch == '\t' ||
		ch == '\n' ||
		ch == '\r' ||
		ch == '\v' ||
		ch == '\f')
		return true;
	return false;
}
#endif

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(tensor_in);
Datum tensor_in(PG_FUNCTION_ARGS) {
	char *string = PG_GETARG_CSTRING(0);
	int32 nitems;
	int32 ndim = 0, shape[TENSOR_MAX_NDIM];
	Node *escontext = fcinfo->context;
	float8 *values;
	char *p = string;
	Tensor *result;
	float8 *data_ptr;
  

	for (int i = 0; i < TENSOR_MAX_NDIM; i++) {
		shape[i] = -1; /* indicates "not yet known" */
	}

	while (tensor_isspace(*p)) {
		p++;
	}

	if (*p != '[')
		ereturn(escontext, (Datum) 0,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("malformed tensor literal: \"%s\"", string),
				 errdetail("Tensor value must start with \"[\"")));
	else {
		if (!ReadTensorStr(&p, &ndim, shape, &nitems, &values, string, escontext))
			return (Datum) 0; //TODO why return 0?

	}

	while (*p) {
		if(!tensor_isspace(*p++))
			ereturn(escontext, (Datum) 0,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("malformed tensor literal: \"%s\"", string),
					 errdetail("Junk after closing square bracket.")));
	}

	if (nitems == 0)
		ereturn(escontext, (Datum) 0,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("empty tensors are not supported")));

	result = InitTensor(ndim, shape);

	data_ptr = TENSOR_DATA(result);
	for (int i = 0; i < nitems; i++) {
		data_ptr[i] = values[i];
	}
  
	pfree(values);

	PG_RETURN_POINTER(result);
}

/*
 * ReadTensorStr :
 *   parses the tensor string pointed to by *srcptr and converts the values
 *   to internal format. Determines the tensor shape as it goes.

 * Outputs:
 *   *ndim_p, shape: dimensions deduced from the input structure
 *   *nitems_p: total number of elements
 *   *values_p: palloc'd array, filled with data values
 *
 * 'origStr' is the original input string, used only in error messages.
 * If *escontext points to an ErrorSaveContext, details of any error are
 * reported there.
 *
 * Result:
 *  true for success, false for failure (if escontext is provided).
 */

static bool ReadTensorStr(char **srcptr,
						  int *ndim_p,
						  int *shape,
						  int *nitems_p,
						  float8 **values_p,
						  const char *origStr,
						  Node *escontext) {

	int ndim = *ndim_p;
	bool dimensions_specified = (ndim != 0);
	int maxitems;
	float8 *values;
	float8 val;
	int nest_level;
	int nitems;
	bool ndim_frozen;
	bool expect_delim;
	int nelems[TENSOR_MAX_NDIM];

	/* Allocate some starting output workspace; enlarged as needed */
  
	maxitems = 16;
	values = palloc_array(float8, maxitems);

	/* Loop below assumes first token is TTOK_LEVEL_START */
	Assert(**srcptr == '[');

	nest_level = 0;
	nitems = 0;
	ndim_frozen = dimensions_specified;
	expect_delim = false;
	do {
		TensorToken tok;

		tok = ReadTensorToken(srcptr, &val, origStr, escontext);

		switch (tok) {
			case TTOK_LEVEL_START:
				/* Can't write left brace where delim is expected */
				if (expect_delim)
					ereturn(escontext, false,
							(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
							 errmsg("malformed tensor literal: \"%s\"", origStr),
							 errdetail("Unexpected \"%c\" character.", '[')));

				/* Initialize element counting in the new level */
				if (nest_level >= TENSOR_MAX_NDIM)
					ereturn(escontext, false,
							(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
							 errmsg("tensor dimensions exceed the maximum allowed (%d)",
									TENSOR_MAX_NDIM)));

				nelems[nest_level] = 0;
				nest_level++;
				if (nest_level > ndim) {
					/* Can't increase ndim once it's frozen */
					if (ndim_frozen)
						goto dimension_error;
					ndim = nest_level;
				}
				break;

			case TTOK_LEVEL_END:
				/* Can't get here with nest_level == 0 */
				Assert(nest_level > 0);

				/*
				 * We allow a right brace to terminate an empty sub-array,
				 * otherwise it must occur where we expect a delimeter.
				 */
				if (nelems[nest_level - 1] > 0 && !expect_delim)
					ereturn(escontext, false,
							(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
							 errmsg("malformed tensor literal: \"%s\"", origStr),
							 errdetail("Unexpected \"%c\" character.", ']')));

				nest_level--;
				/* Nested suub-arrays count as elements of outer level */
				if (nest_level > 0)
					nelems[nest_level - 1]++;

				/* we set the shape after the first subarray of that depth */
				if (shape[nest_level] < 0) {
					shape[nest_level] = nelems[nest_level];
				} else if (nelems[nest_level] != shape[nest_level]) {
					/* subsequent subtensors must have the same length */
					goto dimension_error;
				}
				/*
				 * Must have a deliim or another right brace followinig, unless
				 * we have reached nest_level 0, where this won't matter.
				 */
				expect_delim = true;
				break;

			case TTOK_DELIM:
				if (!expect_delim)
					ereturn(escontext, false,
							(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
							 errmsg("malformed tensor literal: \"%s\"", origStr),
							 errdetail("Unexpected \"%c\" character.", ',')));
				expect_delim = false;
				break;

			case TTOK_ELEM:
				/* Can't get here with nest_level == 0 */
				Assert(nest_level > 0);

				/* Disallow consecutive ELEM tokens */
				if (expect_delim)
					ereturn(escontext, false,
							(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
							 errmsg("malformed tensor literal: \"%s\"", origStr),
							 errdetail("Unexpected tensor element.")));

				/* Enlarge the values array if needed */
				if (nitems >= maxitems) {
					if (maxitems >= TENSOR_MAX_ELEM)
						ereturn(escontext, false,
								(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
								 errmsg("tensor size exceeds the maximum allowed (%d)",
										TENSOR_MAX_ELEM)));

					maxitems = Min(maxitems * 2, TENSOR_MAX_ELEM);
					values = repalloc_array(values, float8, maxitems);
				}

				values[nitems++] = val;

				/* once we've found an element, the number of dimensions can
				 * no longer increase, and subsequent elements must all be at
				 * the same nesting depth.
				 */
				ndim_frozen = true;
				if (nest_level != ndim)
					goto dimension_error;

				/* Count the new element */
				nelems[nest_level - 1]++;

				/* Muust have a delim or right brace followiiig */
				expect_delim = true;
				break;
			case TTOK_ERROR:
				return false;
		}
	} while (nest_level > 0);

	*ndim_p = ndim;
	*nitems_p = nitems;
	*values_p = values;

	return true;

dimension_error:
	ereturn(escontext, false,
			(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
			 errmsg("malformed tensor literal: \"%s\"", origStr),
			 errdetail("Tensors must have subtensors with matching shape.")));
}

static TensorToken
ReadTensorToken(char **srcptr, float8 *val,
				const char *origStr, Node *escontext) {
	char *p = *srcptr;
	char *stringEnd;
	float8 v;

	for (;;) {
		switch (*p) {
			case '\0':
				goto ending_error;
			case '[':
				*srcptr = p + 1;
				return TTOK_LEVEL_START;
			case ']':
				*srcptr = p + 1;
				return TTOK_LEVEL_END;
			default:
				if (*p == ',') {
					*srcptr = p + 1;
					return TTOK_DELIM;
				}
				if (tensor_isspace(*p)) {
					p++;
					continue;
				}

				errno = 0;
				v = strtof(p, &stringEnd);

				if (stringEnd == p)
					ereturn(escontext, false,
							(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
							 errmsg("invalid input syntax for type tensor: \"%s\"", origStr)));

				if (errno == ERANGE && isinf(v))
					ereturn(escontext, false,
							(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
							 errmsg("\"%s\" is out of range for type tensor", pnstrdup(p, stringEnd - p))));

				CheckElement(v);
				*srcptr = stringEnd;
				*val = v;
				return TTOK_ELEM;
      
		}
	}

ending_error:
	ereturn(escontext, TTOK_ERROR,
			(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
			 errmsg("malformed tensor literal \"%s\"", origStr),
			 errdetail("Unexpected end of input.")));
}

#define AppendChar(ptr, c) (*(ptr)++ = (c))
#define AppendFloat(ptr, f) ((ptr) += float_to_shortest_decimal_bufn((f), (ptr)))


FUNCTION_PREFIX PG_FUNCTION_INFO_V1(tensor_out);
Datum tensor_out(PG_FUNCTION_ARGS) {
	Tensor *t = PG_GETARG_TENSOR_P(0);
	int ndim = t->ndim;
	int *shape = t->shape;
	float8 *data_ptr = TENSOR_DATA(t);
	int nitems = tensor_nitems(t);
	// half of total, multiply by 2. starts at 1 for outer
	int ndelims = 1;
	int req_length;
	int i, j, k, indx[TENSOR_MAX_NDIM];
	char *buf;
	char *ptr;


	/*
	 * Need:
	 * nitems * (FLOAT_SHORTEST_DECIMAL_LEN - 1) bytes for
	 * float_to_shortest_decimal_bufn
	 *
	 * nitems - 1 bytes for separators
	 *
	 * for delims, see loop. sum of the prefix
	 * products of the shape array save for the last dim
	 *
	 * 1 for \0
	 */

	for (i = 0, k = 1; i < ndim-1; i++) {
		k *= shape[i];
		ndelims += k;
	}

	req_length = (FLOAT_SHORTEST_DECIMAL_LEN - 1) * nitems + (nitems - 1) + ndelims * 2 + 1;
	buf = (char *) palloc(req_length);
	ptr = buf;

	AppendChar(ptr, '[');
	for (i = 0; i < ndim; i++)
		indx[i] = 0;
	j = 0;
	k = 0;
	do {
		for (i = j; i < ndim - 1; i++)
			AppendChar(ptr, '[');

		AppendFloat(ptr, data_ptr[k++]);
		for (i = ndim - 1; i >= 0; i--) {
			if (++(indx[i]) < shape[i]) {
				AppendChar(ptr, ',');
				break;
			} else {
				indx[i] = 0;
				AppendChar(ptr, ']');
			}
		}
		j = i;
	} while (j != -1);

	*ptr = '\0';
	Assert(overall_length == (ptr - buf + 1));

	PG_FREE_IF_COPY(t, 0);
	PG_RETURN_CSTRING(buf);
}

/*
 * tensor_recv :
 *         converts a tensor from the external binary format to
 *         its internal format.
 * return value :
 *        the internal representation of the input tensor
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(tensor_recv);
Datum
tensor_recv(PG_FUNCTION_ARGS) {
	StringInfo buf = (StringInfo) PG_GETARG_POINTER(0);
	uint8 dtype, flags;
	int i, nitems;
	Tensor *retval;
	int16 ndim;
	int32 shape[TENSOR_MAX_NDIM];
	float8 *data;

	dtype = pq_getmsgint(buf, 1);
	flags = pq_getmsgint(buf, 1);
	ndim = pq_getmsgint(buf, 2);
	if (ndim <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("invalid number of dimensions: %d", ndim)));
	if (ndim > TENSOR_MAX_NDIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("number of tensor dimensions (%d) exceeds the maximum allowed (%d)",
						ndim, TENSOR_MAX_NDIM)));
	for (i = 0; i < ndim; i++) {
		shape[i] = pq_getmsgint(buf, 4);
	}

	nitems = 1;
	for (i = 0; i < ndim; i++) {
		nitems *= shape[i];
	}
	// TODO check overflow of nitems?
	data = (float8 *) palloc(nitems * sizeof(float8));
	for (i = 0; i < nitems; i++) {
		data[i] = pq_getmsgfloat8(buf);
	}
	retval = InitTensor(ndim, shape);
	SET_VARSIZE(retval, tensor_size(ndim, nitems));
	retval->dtype = dtype;
	retval->flags = flags;
	retval->ndim = ndim;
	memcpy(TENSOR_SHAPE(retval), shape, ndim * sizeof(int32));
	memcpy(TENSOR_DATA(retval), data, nitems * sizeof(float8));

	pfree(data);

	PG_RETURN_TENSOR_P(retval);
}
  

/*
 * tensor_send :
 *         takes the internal representation of a tensor and returns a bytea
 *         containing the tensor in its external binary format
 */

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(tensor_send);
Datum
tensor_send(PG_FUNCTION_ARGS) {
	Tensor *t = PG_GETARG_TENSOR_P(0);
	uint8 dtype, flags;
	int nitems, i;
	int32 ndim, *shape;
	float8 *data;
	StringInfoData buf;

	dtype = t->dtype;
	flags = t->flags;
	ndim = TENSOR_NDIM(t);
	shape = TENSOR_SHAPE(t);
	nitems = tensor_nitems(t);
	data = TENSOR_DATA(t);
  
	pq_begintypsend(&buf);

	pq_sendint8(&buf, dtype);
	pq_sendint8(&buf, flags);
	pq_sendint16(&buf, ndim);
	for (i = 0; i < ndim; i++) {
		pq_sendint32(&buf, shape[i]);
	}

	for (i = 0; i < nitems; i++) {
		pq_sendfloat8(&buf, data[i]);
	}

	PG_RETURN_BYTEA_P(pq_endtypsend(&buf));
}

/*
 * Convert array to tensor
 */

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(array_to_tensor);
Datum
array_to_tensor(PG_FUNCTION_ARGS) {
	ArrayType *array = PG_GETARG_ARRAYTYPE_P(0);
	Tensor *result;
	int16 typlen;
	bool typbyval;
	char typalign;
	int i, ndim, *dim, nelemsp;
	float8 *data;
	Datum *elemsp;

	if (ARR_HASNULL(array) && array_contains_nulls(array))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("array must not contain nulls")));

	ndim = ARR_NDIM(array);
	dim = ARR_DIMS(array);

	if (ndim == 0) {
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("empty tensors are not supported")));
	}

	get_typlenbyvalalign(ARR_ELEMTYPE(array), &typlen, &typbyval, &typalign);
	deconstruct_array(array, ARR_ELEMTYPE(array), typlen, typbyval, typalign, &elemsp, NULL, &nelemsp);

	//TODO Check # of elements;

	result = InitTensor(ndim, dim);
	data = TENSOR_DATA(result);

	if (ARR_ELEMTYPE(array) == INT4OID) {
		for (i = 0; i < nelemsp; i++)
			data[i] = DatumGetInt32(elemsp[i]);
	} else if (ARR_ELEMTYPE(array) == FLOAT8OID) {
		for (i = 0; i < nelemsp; i++)
			data[i] = DatumGetFloat8(elemsp[i]);
	} else if (ARR_ELEMTYPE(array) == FLOAT4OID) {
		for (i = 0; i < nelemsp; i++)
			data[i] = DatumGetFloat4(elemsp[i]);
	} else if (ARR_ELEMTYPE(array) == NUMERICOID) {
		for (i = 0; i < nelemsp; i++)
			data[i] = DatumGetFloat8(DirectFunctionCall1(numeric_float8, elemsp[i]));
	} else {
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("unsupported array type")));
	}

	pfree(elemsp);

	/* Ensure non-NaN and non-infinity */
	for (i = 0; i < nelemsp; i++)
		CheckElement(data[i]);

	PG_RETURN_POINTER(result);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(tensor_to_float8);
Datum
tensor_to_float8(PG_FUNCTION_ARGS) {
	Tensor *t = PG_GETARG_TENSOR_P(0);
	Datum *datums;
	ArrayType *result;
	int i, nitems, ndim, *lbs;
	float8 *data = TENSOR_DATA(t);

	ndim = TENSOR_NDIM(t);

	lbs = palloc0(sizeof(int) * ndim);

	nitems = tensor_nitems(t);
	datums = (Datum *) palloc(sizeof(Datum) * nitems);
  
  
	for (i = 0; i < nitems; i++)
		datums[i] = Float8GetDatum(data[i]);

	/*
	 * TODO what are elmbyval and elmalign
	 */
	result = construct_md_array(datums,
		       		    NULL,
				    TENSOR_NDIM(t),
				    TENSOR_SHAPE(t),
				    lbs,
				    FLOAT8OID,
				    sizeof(float8),
				    true,
				    TYPALIGN_INT);

	pfree(datums);
	pfree(lbs);
	PG_RETURN_POINTER(result);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(tensor_add);
Datum
tensor_add(PG_FUNCTION_ARGS)
{
	Tensor	   *a = PG_GETARG_TENSOR_P(0);
	Tensor	   *b = PG_GETARG_TENSOR_P(1);
	float8	   *ax = (float8 *)(((char *) a) + a->data_offset);
	float8	   *bx = (float8 *)(((char *) b) + b->data_offset);
	int          n = tensor_nitems(a);
	Tensor	   *result;
	float8	   *rx;

	CheckShapes(a, b);

	result = InitTensor(a->ndim, a->shape);
	rx = (float8 *)(((char *) result) + result->data_offset);

	for (int i = 0, imax = n; i < imax; i++) {
		rx[i] = ax[i] + bx[i];
	}

	/* Check for overflow */
	for (int i = 0, imax = n; i < imax; i++)
	{
		if (isinf(rx[i]))
			float_overflow_error();
	}

	PG_RETURN_POINTER(result);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(tensor_sub);
Datum
tensor_sub(PG_FUNCTION_ARGS)
{
	Tensor	   *a = PG_GETARG_TENSOR_P(0);
	Tensor	   *b = PG_GETARG_TENSOR_P(1);
	float8	   *ax = (float8 *)(((char *) a) + a->data_offset);
	float8	   *bx = (float8 *)(((char *) b) + b->data_offset);
	int          n = tensor_nitems(a);
	Tensor	   *result;
	float8	   *rx;

	CheckShapes(a, b);

	result = InitTensor(a->ndim, a->shape);
	rx = (float8 *)(((char *) result) + result->data_offset);

	for (int i = 0, imax = n; i < imax; i++) {
		rx[i] = ax[i] - bx[i];
	}

	/* Check for overflow */
	for (int i = 0, imax = n; i < imax; i++)
	{
		if (isinf(rx[i]))
			float_overflow_error();
	}

	PG_RETURN_POINTER(result);
}

