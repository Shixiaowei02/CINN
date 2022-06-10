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

#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/pass/pass_test_helper.h"
#include "cinn/runtime/cuda/cuda_util.h"
#include "gtest/gtest.h"

namespace cinn::frontend::pass {

/*
 * DotMerger Test
 *
 * Before:
 * (m, k) * (k, n1) -> (m1, n1)  ==> (m, n1 + n2)
 * (m, k) * (k, n2) -> (m2, n2)
 *
 * After:
 * (k, n1) concat (k, n2) -> (k, n1 + n2)
 * (m, k) * (k, n1 + n2) -> (m, n1 + n2)
 * (m, n1 + n2) slice -> (m, n1), (m, n2)
 */

TEST(DotMerger, lhs) {
  if (!IsCompiledWithCUDA()) {
    // because op def changes with the macro
    return;
  }
  cinn::runtime::cuda::CublasHandle::get_instance();
  int m = 2, k = 10201, n1 = 50, n2 = 50, axis = 1;
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {m, k}, "A");
  auto b = builder.CreateInput(Float(32), {k, n1}, "B");
  auto c = builder.CreateInput(Float(32), {k, n2}, "C");
  auto d = builder.Matmul(a, b);
  auto e = builder.Matmul(a, c);
  auto f = builder.CreateInput(Float(32), {m, n1}, "D");
  auto g = builder.Add(d, f);
  auto h = builder.Add(e, g);
  auto p = builder.Build();

  Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), b.id(), c.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{
      {"Decomposer", "RemoveIdentity"}, {"TransposeFoldingInput", "DotMerger", "GemmRewriter"}};
  CompareResult(&p, target, input_ids, {h->id}, -2, std::move(passes), 123, true);
}

/*
 * DotMerger Test
 *
 * Before:
 * (m1, k) * (k, n) -> (m1, n)  ==> (m1 + m2, n)
 * (m2, k) * (k, n) -> (m2, n)
 *
 * After:
 * (m1, k) concat (m2, k) -> (m1 + m2, k)
 * (m1 + m2, k) * (k, n) -> (m1 + m2, n)
 * (m1 + m2, n) slice -> (m1, n), (m2, n)
 */
/*
TEST(DotMerger, rhs) {
  if (!IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  int m1 = 50, m2 = 50, k = 10201, n = 2, axis = 0;
  auto a = builder.CreateInput(Float(32), {m1, k}, "A");
  auto b = builder.CreateInput(Float(32), {m2, k}, "B");
  auto c = builder.CreateInput(Float(32), {k, n}, "C");
  auto d = builder.Matmul(a, c);
  auto e = builder.Matmul(b, c);
  auto f = builder.Concat({d, e}, axis);
  auto p = builder.Build();

  Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), b.id(), c.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  std::pair<std::vector<std::string>, std::vector<std::string>> passes{{"Decomposer", "RemoveIdentity"},
                                                                       {"TransposeFoldingInput", "DotMerger"}};
  CompareResult(&p, target, input_ids, {f->id}, -2, std::move(passes), 123, true);
}
*/
}  // namespace cinn::frontend::pass