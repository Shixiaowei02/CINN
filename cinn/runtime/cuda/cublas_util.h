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
#include "glog/logging.h"

namespace cinn {
namespace runtime {
namespace cuda {

cublasStatus_t cublasGemm(cudaDataType_t dtype,
                          cublasHandle_t handle,
                          cublasOperation_t transa,
                          cublasOperation_t transb,
                          int m,
                          int n,
                          int k,
                          float *alpha,
                          const void *A,
                          int lda,
                          const void *B,
                          int ldb,
                          float *beta,
                          void *C,
                          int ldc) {
  if (dtype == CUDA_R_32F) {
    LOG(INFO) << "cublasSgemm";
    return cublasSgemm(handle,
                       transa,
                       transb,
                       m,
                       n,
                       k,
                       reinterpret_cast<const float *>(alpha),
                       reinterpret_cast<const float *>(A),
                       lda,
                       reinterpret_cast<const float *>(B),
                       ldb,
                       reinterpret_cast<const float *>(beta),
                       reinterpret_cast<float *>(C),
                       ldc);
  } else if (dtype == CUDA_R_16F) {
    LOG(INFO) << "cublasHgemm";
    return {};
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
                                        float *alpha,
                                        const void *A,
                                        int lda,
                                        long long int strideA,
                                        const void *B,
                                        int ldb,
                                        long long int strideB,
                                        float *beta,
                                        void *C,
                                        int ldc,
                                        long long int strideC,
                                        int batchCount) {
  if (dtype == CUDA_R_32F) {
    LOG(INFO) << "cublasSgemmStridedBatched";
    return cublasSgemmStridedBatched(handle,
                                     transa,
                                     transb,
                                     m,
                                     n,
                                     k,
                                     reinterpret_cast<const float *>(alpha),
                                     reinterpret_cast<const float *>(A),
                                     lda,
                                     strideA,
                                     reinterpret_cast<const float *>(B),
                                     ldb,
                                     strideB,
                                     reinterpret_cast<const float *>(beta),
                                     reinterpret_cast<float *>(C),
                                     ldc,
                                     strideC,
                                     batchCount);
  } else if (dtype == CUDA_R_16F) {
    LOG(INFO) << "cublasHgemmStridedBatched";
    return {};
  }
  LOG(FATAL) << "Unsupported cublasGemmStridedBatched precision.";
}

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
