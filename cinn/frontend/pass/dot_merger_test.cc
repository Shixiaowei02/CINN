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

  ProgramPass::Apply(program, fetch_ids, target, {"TransposeFolding", "DotMerger"});

  auto fused_out = RunProgram(*program, target, input_ids, output_ids, seed, print_tensor);

  ASSERT_EQ(origin_out.size(), fused_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out[i], fused_out[i]);
  }
}
}  // namespace

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

TEST(DotMerger, lhs) {
  if (!IsCompiledWithCUDA()) {
    return;
  }
  NetBuilder builder("net_builder");
  auto a       = builder.CreateInput(Float(32), {2, 1}, "A");
  auto b       = builder.CreateInput(Float(32), {1, 2}, "B");
  auto c       = builder.CreateInput(Float(32), {1, 1}, "C");
  auto d       = builder.Matmul(a, b);  // {2, 2}
  auto e       = builder.Matmul(a, c);  // {2, 1}
  auto f       = builder.Concat({d, e}, 1);
  auto program = builder.Build();

  Target target = common::DefaultNVGPUTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), b.id(), c.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  CompareResult(&program, target, input_ids, {f->id}, 0, 123, true);
}

}  // namespace cinn::frontend::pass
