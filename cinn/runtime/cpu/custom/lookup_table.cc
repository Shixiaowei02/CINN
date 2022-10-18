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

#include "cinn/runtime/cpu/custom/lookup_table.h"

#include "glog/logging.h"

namespace cinn {
namespace runtime {
namespace cpu {
namespace custom {

template <typename T>
void lookup_table(T* output,
                  const T* table,
                  const int64_t* ids,
                  int64_t row_number,
                  int64_t row_width,
                  int64_t ids_numel,
                  int64_t padding_idx) {
  constexpr int64_t kNoPadding = -1;
  for (int64_t i = 0; i < ids_numel; ++i) {
    auto* dst = output + i * row_width;
    if (padding_idx != kNoPadding && ids[i] == padding_idx) {
      std::fill_n(dst, row_width, 0);
    } else {
      CHECK_LT(ids[i], row_number);
      CHECK_GE(ids[i], 0);
      const auto* src = table + ids[i] * row_width;
      std::copy_n(src, row_width, dst);
    }
  }
}

template void lookup_table<float>(float* output,
                                  const float* table,
                                  const int64_t* ids,
                                  int64_t row_number,
                                  int64_t row_width,
                                  int64_t ids_numel,
                                  int64_t padding_idx);

}  // namespace custom
}  // namespace cpu
}  // namespace runtime
}  // namespace cinn
