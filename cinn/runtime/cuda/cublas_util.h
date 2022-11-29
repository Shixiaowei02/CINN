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
  int a_size = m * n;
  int b_size = n * k;
  util::Vector<T> tmpA(const_cast<T *>(A), a_size);
  std::vector<T> tmpA_host = tmpA.to_host();
  for (int i = 0; i < a_size; ++i) {
    std::cout << tmpA_host[i] << ", ";
  }
  std::cout << '\n';
  util::Vector<T> tmpB(const_cast<T *>(B), b_size);
  std::vector<T> tmpB_host = tmpB.to_host();
  for (int i = 0; i < b_size; ++i) {
    std::cout << tmpB_host[i] << ", ";
  }
  std::cout << '\n';
  int c_size = m * k;
  util::Vector<T> tmpC(const_cast<T *>(C), c_size);
  std::vector<T> tmpC_host = tmpC.to_host();
  for (int i = 0; i < c_size; ++i) {
    std::cout << tmpC_host[i] << ", ";
  }
  std::cout << '\n';
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
    LOG(INFO) << "--- CUDA_R_32F";
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
    /*
    LOG(INFO) << "transa = " << transa << ", "
              << "transb = " << transb << ", "
              << "m = " << m << ", "
              << "n = " << n << ", "
              << "k = " << k << ", "
              << "lda = " << lda << ", "
              << "ldb = " << ldb << ", "
              << "ldc = " << ldc;
    cudaDeviceSynchronize();
    debug_str(m,
              n,
              k,
              reinterpret_cast<const common::float16 *>(A),
              reinterpret_cast<const common::float16 *>(B),
              reinterpret_cast<const common::float16 *>(C));
    */
    return ret;
  }
  LOG(FATAL) << "Unsupported cublasGemm precision.";
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
