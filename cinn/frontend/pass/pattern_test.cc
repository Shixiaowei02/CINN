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

#include "cinn/frontend/pass/pattern.h"

#include "cinn/frontend/pass/test_utils.h"
#include "gtest/gtest.h"

namespace cinn::frontend::pass {

TEST(Pattern, match) {
  auto generate_src_pattern = []() -> Pattern {
    Pattern pattern;
    auto* input_0  = pattern.AddVar()->set_external(true);
    auto* input_1  = pattern.AddVar()->set_external(true);
    auto* input_2  = pattern.AddVar()->set_external(true);
    auto* output_0 = pattern.AddVar()->set_external(true);
    auto* output_1 = pattern.AddVar()->set_external(true);

    auto* matmul_0 =
        pattern.AddInstr("matmul", std::vector<VarRepr*>{input_0, input_2}, std::vector<VarRepr*>{output_0});
    auto* matmul_1 =
        pattern.AddInstr("matmul", std::vector<VarRepr*>{input_0, input_1}, std::vector<VarRepr*>{output_1});

    CHECK_EQ(pattern.cur_id(), 6);
    CHECK_EQ(pattern.nodes().size(), 7u);
    return pattern;
  };

  auto generate_program = []() -> Program {
    NetBuilder builder("net_builder");
    auto a       = builder.CreateInput(Float(32), {10201, 50}, "A");
    auto b       = builder.CreateInput(Float(32), {50, 50}, "B");
    auto c       = builder.CreateInput(Float(32), {50, 50}, "C");
    auto d       = builder.Matmul(a, b);
    auto e       = builder.Matmul(a, c);
    auto program = builder.Build();
    return program;
  };
}

}  // namespace cinn::frontend::pass
