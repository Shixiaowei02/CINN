/**
 * \file This file contains the reduce intrinsics available to be used in CUDA code generated by CodeGen.
 */

#define EXPAND_REDUCE_FP32_MACRO(MACRO, ...) \
  MACRO(sum_fp32, 0.0f, float, ##__VA_ARGS__) \
  MACRO(prod_fp32, 1.0f, float, ##__VA_ARGS__) \
  MACRO(max_fp32, 3.402823e+38f, float, ##__VA_ARGS__) \
  MACRO(min_fp32, -3.402823e+38f, float, ##__VA_ARGS__) \

#ifdef FN_FP32
#define FUNC_FP32(func) FN_FP32(func)
#else
#define FUNC_FP32(func) cinn_nvgpu_##func##_fp32
#endif

__device__ inline float cinn_sum_fp32(const float left, const float right) { return left + right; }
__device__ inline float cinn_prod_fp32(const float left, const float right) { return left * right; }
__device__ inline float cinn_max_fp32(const float left, const float right) { return FUNC_FP32(max)(left, right); }
__device__ inline float cinn_min_fp32(const float left, const float right) { return FUNC_FP32(min)(left, right); }


#ifdef CINN_CUDA_FP16

#define EXPAND_REDUCE_FP16_MACRO(MACRO, ...) \
  MACRO(sum_fp16, 0.0f, float16, ##__VA_ARGS__) \
  MACRO(prod_fp16, 1.0f, float16, ##__VA_ARGS__) \
  MACRO(max_fp16, -65504f, float16, ##__VA_ARGS__) \
  MACRO(min_fp16, 65504f, float16, ##__VA_ARGS__)

#ifdef FN_FP16
#define FUNC_FP16(func) FN_FP16(func)
#else
#define FUNC_FP16(func) cinn_nvgpu_##func##_fp16
#endif

__device__ inline float16 cinn_sum_fp16(const float16 left, const float16 right) { return left + right; }
__device__ inline float16 cinn_prod_fp16(const float16 left, const float16 right) { return left * right; }
__device__ inline float16 cinn_max_fp16(const float16 left, const float16 right) { return FUNC_FP16(max)(left, right); }
__device__ inline float16 cinn_min_fp16(const float16 left, const float16 right) { return FUNC_FP16(min)(left, right); }
#endif


#define EXPAND_REDUCE_BOOL_MACRO(MACRO, ...) \
  MACRO(all, true, bool, ##__VA_ARGS__) \
  MACRO(any, false, bool, ##__VA_ARGS__)

__device__ inline bool cinn_all(const bool left, const bool right) { return left && right; }
__device__ inline bool cinn_any(const bool left, const bool right) { return left || right; }


#define CINN_SHUFFLE_FUNCTION(offset, op, init)                  \
  shfl_res = __shfl_down_sync(mask, tmp_val, offset, 32);        \
  shfl_res = threadIdx.x % 32 + offset < lane ? shfl_res : init; \
  tmp_val  = op(tmp_val, shfl_res);


#define CINN_WARP_SHUFFLE_INTERNAL_IMPL(REDUCE_TYPE, INITIAL_VALUE, DTYPE) \
__device__ inline DTYPE cinn_warp_shuffle_##REDUCE_TYPE##_internal(const DTYPE value) { \
  DTYPE tmp_val      = value, shfl_res; \
  unsigned int mask = __activemask(); \
  unsigned int lane = __popc(mask); \
  CINN_SHUFFLE_FUNCTION(16, cinn_##REDUCE_TYPE, DTYPE(INITIAL_VALUE)) \
  CINN_SHUFFLE_FUNCTION(8, cinn_##REDUCE_TYPE, DTYPE(INITIAL_VALUE)) \
  CINN_SHUFFLE_FUNCTION(4, cinn_##REDUCE_TYPE, DTYPE(INITIAL_VALUE)) \
  CINN_SHUFFLE_FUNCTION(2, cinn_##REDUCE_TYPE, DTYPE(INITIAL_VALUE)) \
  CINN_SHUFFLE_FUNCTION(1, cinn_##REDUCE_TYPE, DTYPE(INITIAL_VALUE)) \
  tmp_val = __shfl_sync(mask, tmp_val, 0, 32); \
  return tmp_val; \
}

EXPAND_REDUCE_FP32_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)
EXPAND_REDUCE_BOOL_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)
#endif

#undef CINN_WARP_SHUFFLE_INTERNAL_IMPL

#define CINN_WARP_REDUCE_IMPL(REDUCE_TYPE, INITIAL_VALUE, DTYPE) \
__device__ inline DTYPE cinn_warp_reduce_##REDUCE_TYPE(const DTYPE *buf, int offset, int extend) { \
  float tmp_val = DTYPE(INITIAL_VALUE); \
  for (int i = threadIdx.x; i < extend; i += 32) { \
    tmp_val = cinn_##REDUCE_TYPE(tmp_val, buf[offset + i]); \
  } \
  return cinn_warp_shuffle_##REDUCE_TYPE##_internal(tmp_val); \
}

EXPAND_REDUCE_FP32_MACRO(CINN_WARP_REDUCE_IMPL)
EXPAND_REDUCE_BOOL_MACRO(CINN_WARP_REDUCE_IMPL)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(CINN_WARP_REDUCE_IMPL)
#endif

#undef CINN_WARP_REDUCE_IMPL

__device__ inline float cinn_warp_reduce_avg_fp32(const float *buf, int offset, int extend) {
  return cinn_warp_reduce_sum_fp32(buf, offset, extend) / extend;
}

#define CINN_BLOCK_REDUCE_INTERNAL_IMPL(TYPE, value, init_value, cinn_warp_shuffle_internal) \
  int warp_id = threadIdx.x / 32;                                                              \
  __shared__ TYPE tmp[32];                                                                     \
  if (warp_id == 0) {                                                                          \
    tmp[threadIdx.x] = init_value;                                                             \
  }                                                                                            \
  TYPE tmp_val = cinn_warp_shuffle_internal(value);                                            \
  if (blockDim.x <= 32) {                                                                      \
    return tmp_val;                                                                            \
  }                                                                                            \
  __syncthreads();                                                                             \
  if (threadIdx.x % 32 == 0) {                                                                 \
    tmp[warp_id] = tmp_val;                                                                    \
  }                                                                                            \
  __syncthreads();                                                                             \
  if (warp_id == 0) {                                                                          \
    tmp_val = tmp[threadIdx.x];                                                                \
    tmp_val = cinn_warp_shuffle_internal(tmp_val);                                             \
    if (threadIdx.x == 0) {                                                                    \
      tmp[0] = tmp_val;                                                                        \
    }                                                                                          \
  }                                                                                            \
  __syncthreads();                                                                             \
  return tmp[0];

#define CINN_BLOCK_REDUCE_INTERNAL_MACRO(REDUCE_TYPE, INITIAL_VALUE, DTYPE) \
__device__ inline DTYPE cinn_block_reduce_##REDUCE_TYPE##_internal(const DTYPE value) { \
  CINN_BLOCK_REDUCE_INTERNAL_IMPL(DTYPE, value, DTYPE(INITIAL_VALUE), cinn_warp_shuffle_##REDUCE_TYPE##_internal); \
}

EXPAND_REDUCE_FP32_MACRO(CINN_BLOCK_REDUCE_INTERNAL_MACRO)
EXPAND_REDUCE_BOOL_MACRO(CINN_BLOCK_REDUCE_INTERNAL_MACRO)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(CINN_BLOCK_REDUCE_INTERNAL_MACRO)
#endif

#undef CINN_BLOCK_REDUCE_INTERNAL_IMPL
#undef CINN_BLOCK_REDUCE_INTERNAL_MACRO

#define CINN_BLOCK_REDUCE_IMPL(REDUCE_TYPE, INITIAL_VALUE, DTYPE) \
__device__ inline DTYPE cinn_block_reduce_##REDUCE_TYPE(const DTYPE *buf, int offset, int extend) { \
  DTYPE tmp_val = DTYPE(INITIAL_VALUE); \
  for (int i = threadIdx.x; i < extend; i += blockDim.x) { \
    tmp_val = cinn_##REDUCE_TYPE(tmp_val, buf[offset + i]); \
  } \
  return cinn_block_reduce_##REDUCE_TYPE##_internal(tmp_val); \
}

EXPAND_REDUCE_FP32_MACRO(CINN_BLOCK_REDUCE_IMPL)
EXPAND_REDUCE_BOOL_MACRO(CINN_BLOCK_REDUCE_IMPL)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(CINN_BLOCK_REDUCE_IMPL)
#endif

#undef CINN_BLOCK_REDUCE_IMPL

#define BLOCK_SHUFFLE_IMPL(REDUCE_TYPE, INITIAL_VALUE, DTYPE) \
__device__ inline DTYPE block_shuffle_##REDUCE_TYPE(const DTYPE *buf, int line, int stride) { \
  DTYPE val = DTYPE(INITIAL_VALUE); \
  for (int idx = threadIdx.x; idx < line; idx += stride) { \
    val = cinn_##REDUCE_TYPE(val, buf[idx]); \
  } \
  return val; \
}

EXPAND_REDUCE_FP32_MACRO(BLOCK_SHUFFLE_IMPL)
EXPAND_REDUCE_BOOL_MACRO(BLOCK_SHUFFLE_IMPL)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(BLOCK_SHUFFLE_IMPL)
#endif

#undef BLOCK_SHUFFLE_IMPL

#undef EXPAND_REDUCE_FP32_MACRO
#undef EXPAND_REDUCE_BOOL_MACRO
#undef EXPAND_REDUCE_FP16_MACRO

#undef FUNC_FP32
#undef FUNC_FP16
