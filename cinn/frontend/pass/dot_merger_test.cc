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

#include "cinn/frontend/pass/test_utils.h"
#include "gtest/gtest.h"

namespace cinn::frontend::pass {
namespace {
void CompareResult(Program* program,
                   const Target& target,
                   const std::vector<std::string>& input_ids,
                   const std::vector<std::string>& output_ids,
                   size_t size_diff,
                   int seed          = -1,
                   bool print_tensor = false) {
  std::unordered_set<std::string> fetch_ids(output_ids.begin(), output_ids.end());
  // apply common pass
  ProgramPass::Apply(program, fetch_ids, target, {"Decomposer", "RemoveIdentity"});

  // get original program size
  auto origin_size = program->size();
  // get original output
  auto origin_out = RunProgram(*program, target, input_ids, output_ids, seed, print_tensor);

  // fuse transpose + add + dot, then run and get the fused output
  ProgramPass::Apply(program, fetch_ids, target, {"TransposeFolding", "GemmRewriter"});

  // get fused program size
  auto fused_size = program->size();
  ASSERT_EQ(size_diff, origin_size - fused_size);
  // get fused output
  auto fused_out = RunProgram(*program, target, input_ids, output_ids, seed, print_tensor);

  ASSERT_EQ(origin_out.size(), fused_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out[i], fused_out[i]);
  }
}
}  // namespace

/*
 * DotMerger Test
 * 2-d Tensor
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

TEST(DotMerger, test) {
  if (!IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
}

}  // namespace cinn::frontend::pass
