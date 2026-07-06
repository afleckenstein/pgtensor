/*-------------------------------------------------------------------------
 *
 * tensor.h
 *    Declarations for Postgres tensors.
 *
 * Much of the structure and code (for parsing, (de)serializing, etc.)
 * is adapted from Postgres arrays.
 *
 * A standard varlena tensor has the following internal structure:
 *    <vl_len_>     - standard varlena header word
 *    <dtype>       - datatype of elements (currently just float8/double)
 *    <flags>       - reserved for future use
 *    <ndim>        - number of dimensions (or "order", or "degree")
 *    <shape>       - length of each axis (C array of int)
 *    <actual data> - whatever is the stored data
 *
 * The <shape> array has ndim elements.
 */
#ifndef TENSOR_H
#define TENSOR_H

#define TENSOR_MAX_NDIM 64
#define TENSOR_MAX_ELEM 16000

#define DatumGetTensor(x) ((Tensor *) PG_DETOAST_DATUM(x))
#define PG_GETARG_TENSOR_P(x) DatumGetTensor(PG_GETARG_DATUM(x))
#define PG_RETURN_TENSOR_P(x) PG_RETURN_POINTER(x)

#define TENSOR_DTYPE(t) ((t)->dtype)
#define TENSOR_FLAGS(t) ((t)->flags)
#define TENSOR_NDIM(t) ((t)->ndim)
#define TENSOR_SHAPE(t) ((t)->shape)
#define TENSOR_DATA(t) ((float8 *) (((char *) (t)) + (t)->data_offset))

/* N.B. every field in this struct is aligned properly */
typedef struct Tensor {
	int32 vl_len_; /* postgres varlena header, don't touch */
	uint8 dtype;
	uint8 flags;
	int16 ndim;
	int32 data_offset;
	int32 shape[FLEXIBLE_ARRAY_MEMBER];
	/* data follows */
} Tensor;

Tensor *InitTensor(int ndim, int shape[]);
void PrintTensor(char *msg, Tensor *tensor);
int tensor_cmp_internal(Tensor *a, Tensor *b);

/* TODO Move to better place */
#if PG_VERSION_NUM >= 160000
#define FUNCTION_PREFIX
#else
#define FUNCTION_PREFIX PGDLLEXPORT
#endif


#endif
