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

/*
TEST(DotMerger, lhs) {
  int m = 2, k = 2, n1 = 2, n2 = 2, n3 = 2, axis = 1;
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {m, k}, "A");
  auto b = builder.CreateInput(Float(32), {k, n1}, "B");
  auto c = builder.Add(a, b);
  auto d = builder.Cast(c, "float16");
  auto p = builder.Build();

  // Target target = common::DefaultNVGPUTarget();
  Target target = common::DefaultHostTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id(), b.id()},
                    std::back_inserter(input_ids),
                    [](absl::string_view id) { return std::string(id); });
  OptimizeConfig passes({{"Decomposer", "RemoveIdentity", "TransposeFoldingInput"}, {}},
                        {{"OpFusionPass", "FusionMergePass"}, {"DotMerger", "OpFusionPass", "FusionMergePass"}});
  CompareResult(&p, target, input_ids, {d->id}, 0, std::move(passes), 123, true);
  LOG(INFO) << "Finished.";
}
*/

TEST(DotMerger, lhs) {
  int m = 2, k = 2, n1 = 2, n2 = 2, n3 = 2, axis = 1;
  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {m, k}, "A");
  auto d = builder.Cast(a, "int64");
  auto p = builder.Build();

  // Target target = common::DefaultNVGPUTarget();
  Target target = common::DefaultHostTarget();
  std::vector<std::string> input_ids;
  absl::c_transform(std::vector<absl::string_view>{a.id()}, std::back_inserter(input_ids), [](absl::string_view id) {
    return std::string(id);
  });
  OptimizeConfig passes({{"Decomposer", "RemoveIdentity", "TransposeFoldingInput"}, {}},
                        {{"OpFusionPass", "FusionMergePass"}, {"DotMerger", "OpFusionPass", "FusionMergePass"}});
  CompareResult(&p, target, input_ids, {d->id}, 0, std::move(passes), 123, true);
  LOG(INFO) << "Finished.";
}

}  // namespace cinn::frontend::pass
