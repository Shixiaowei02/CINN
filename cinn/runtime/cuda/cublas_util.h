// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cublas_v2.h>

#include "cinn/common/type.h"
#include "cinn/runtime/cuda/test_util.h"
#include "glog/logging.h"

namespace cinn {
namespace runtime {
namespace cuda {

template <typename T>
void debug_str(int m, int n, int k, const T *A, const T *B, const T *C) {
  {
    int a_size = m * n;
    util::Vector<T> tmpA(const_cast<T *>(A), a_size);
    std::vector<T> tmpA_host = tmpA.to_host();
    float sum{0};
    for (auto i : tmpA_host) {
      sum += static_cast<float>(i);
    }
    float average = sum / tmpA_host.size();
    LOG(INFO) << "A average: " << average;
  }
  {
    int b_size = n * k;
    util::Vector<T> tmpB(const_cast<T *>(B), b_size);
    std::vector<T> tmpB_host = tmpB.to_host();
    float sum{0};
    for (auto i : tmpB_host) {
      sum += static_cast<float>(i);
    }
    float average = sum / tmpB_host.size();
    LOG(INFO) << "B average: " << average;
  }
  {
    int c_size = m * k;
    util::Vector<T> tmpC(const_cast<T *>(C), c_size);
    std::vector<T> tmpC_host = tmpC.to_host();
    float sum{0};
    for (auto i : tmpC_host) {
      sum += static_cast<float>(i);
    }
    float average = sum / tmpC_host.size();
    LOG(INFO) << "C average: " << average;
  }
}

cublasStatus_t cublasGemm(cudaDataType_t dtype,
                          cublasHandle_t handle,
                          cublasOperation_t transa,
                          cublasOperation_t transb,
                          int m,
                          int n,
                          int k,
                          float alpha,
                          const void *A,
                          int lda,
                          const void *B,
                          int ldb,
                          float beta,
                          void *C,
                          int ldc) {
  if (dtype == CUDA_R_32F) {
    LOG(INFO) << "cublasSgemm f32";
    return cublasSgemm(handle,
                       transa,
                       transb,
                       m,
                       n,
                       k,
                       reinterpret_cast<const float *>(&alpha),
                       reinterpret_cast<const float *>(A),
                       lda,
                       reinterpret_cast<const float *>(B),
                       ldb,
                       reinterpret_cast<const float *>(&beta),
                       reinterpret_cast<float *>(C),
                       ldc);
  } else if (dtype == CUDA_R_16F) {
    LOG(INFO) << "cublasSgemm f16";
    common::float16 alpha_fp16{alpha};
    common::float16 beta_fp16{beta};
    auto ret = cublasHgemm(handle,
                           transa,
                           transb,
                           m,
                           n,
                           k,
                           reinterpret_cast<const __half *>(&alpha_fp16),
                           reinterpret_cast<const __half *>(A),
                           lda,
                           reinterpret_cast<const __half *>(B),
                           ldb,
                           reinterpret_cast<const __half *>(&beta_fp16),
                           reinterpret_cast<__half *>(C),
                           ldc);
    debug_str(m,
              n,
              k,
              reinterpret_cast<const common::float16 *>(A),
              reinterpret_cast<const common::float16 *>(B),
              reinterpret_cast<const common::float16 *>(C));
    return ret;
  }
  LOG(FATAL) << "Unsupported cublasGemm precision.";
  return {};
}

cublasStatus_t cublasGemmStridedBatched(cudaDataType_t dtype,
                                        cublasHandle_t handle,
                                        cublasOperation_t transa,
                                        cublasOperation_t transb,
                                        int m,
                                        int n,
                                        int k,
                                        float alpha,
                                        const void *A,
                                        int lda,
                                        long long int strideA,
                                        const void *B,
                                        int ldb,
                                        long long int strideB,
                                        float beta,
                                        void *C,
                                        int ldc,
                                        long long int strideC,
                                        int batchCount) {
  if (dtype == CUDA_R_32F) {
    LOG(INFO) << "cublasHgemmStridedBatched FP32";
    return cublasSgemmStridedBatched(handle,
                                     transa,
                                     transb,
                                     m,
                                     n,
                                     k,
                                     reinterpret_cast<const float *>(&alpha),
                                     reinterpret_cast<const float *>(A),
                                     lda,
                                     strideA,
                                     reinterpret_cast<const float *>(B),
                                     ldb,
                                     strideB,
                                     reinterpret_cast<const float *>(&beta),
                                     reinterpret_cast<float *>(C),
                                     ldc,
                                     strideC,
                                     batchCount);
  } else if (dtype == CUDA_R_16F) {
    LOG(INFO) << "cublasHgemmStridedBatched FP16";
    return cublasHgemmStridedBatched(handle,
                                     transa,
                                     transb,
                                     m,
                                     n,
                                     k,
                                     reinterpret_cast<const __half *>(&alpha),
                                     reinterpret_cast<const __half *>(A),
                                     lda,
                                     strideA,
                                     reinterpret_cast<const __half *>(B),
                                     ldb,
                                     strideB,
                                     reinterpret_cast<const __half *>(&beta),
                                     reinterpret_cast<__half *>(C),
                                     ldc,
                                     strideC,
                                     batchCount);
  }
  LOG(FATAL) << "Unsupported cublasGemmStridedBatched precision.";
}

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
