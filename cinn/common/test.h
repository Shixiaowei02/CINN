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

#pragma once
/*
#include <climits>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
*/
// The `HOST` macro definition is not used here, it has a potential
// conflict with the enumeration `kHOST` representing the backend.
//#define __host__
//#define __device__

namespace cinn {
namespace common {

// Use CINN_ALIGNED(2) to ensure that each float16 will be allocated
// and aligned at least on a 2-byte boundary, which leads to efficient
// memory access of float16 struct and also makes float16 compatible
// with CUDA half
struct float16111 {
 public:
  int x;
  /*
    // The following defaulted special class member functions
    // are added to make float16 pass the std::is_trivial test
    float16()                 = default;
    float16(const float16& o) = default;
    float16& operator=(const float16& o) = default;
    float16(float16&& o)                 = default;
    float16& operator=(float16&& o) = default;
    ~float16()                      = default;
  */
};

}  // namespace common
}  // namespace cinn
