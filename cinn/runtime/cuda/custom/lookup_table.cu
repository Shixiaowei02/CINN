// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

namespace cinn {
namespace runtime {
namespace cuda {
namespace custom {

namespace {
template <int BlockDimX, int BlockDimY, int GridDimX, typename T>
__global__ void LookupTableKernel(T* output,
                                  const T* table,
                                  const int64_t* ids,
                                  const int64_t N,
                                  const int64_t K,
                                  const int64_t D,
                                  const int64_t padding_idx) {
  constexpr int64_t kNoPadding = -1;
  int idx                      = threadIdx.x;
  int idy                      = blockIdx.x + threadIdx.y * GridDimX;

  while (idy < K) {
    int64_t id   = ids[idy];
    T* out       = output + idy * D;
    const T* tab = table + id * D;
    for (int i = idx; i < D; i += BlockDimX) {
      if (padding_flag != kNoPadding && id == padding_idx) {
        out[i] = 0;
      } else {
        out[i] = tab[i];
      }
    }
    idy += BlockDimY * GridDimX;
  }
}
}  // namespace

template <typename T>
void lookup_table(T* output,
                  const T* table,
                  const int64_t* ids,
                  int64_t row_number,
                  int64_t row_width,
                  int64_t ids_numel,
                  int64_t padding_idx,
                  cudaStream_t stream) {
  dim3 threads(128, 8);
  dim3 grids(8, 1);
  LookupTableKernel<128, 8, 8>
      <<<grids, threads, 0, stream>>>(output, table, ids, row_number, row_width, ids_numel, padding_idx);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    LOG(INFO) << cudaGetErrorString(error);
  }
}

template void lookup_table<float>(float* output,
                                  const float* table,
                                  const int64_t* ids,
                                  int64_t row_number,
                                  int64_t row_width,
                                  int64_t ids_numel,
                                  int64_t padding_idx,
                                  cudaStream_t stream);

}  // namespace custom
}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
