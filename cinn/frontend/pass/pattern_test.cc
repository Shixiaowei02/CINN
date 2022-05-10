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

#include "cinn/frontend/net_builder.h"
#include "gtest/gtest.h"

namespace cinn::frontend::pass {

TEST(Pattern, match) {
  auto generate_src_pattern = []() -> Digraph {
    PatternBuilder builder;
    auto* input_0  = builder.AddVar();
    auto* input_1  = builder.AddVar();
    auto* input_2  = builder.AddVar();
    auto* output_0 = builder.AddVar();
    auto* output_1 = builder.AddVar();

    auto* matmul_0 = builder.AddInstr(
        "matmul", std::vector<PatternVar const*>{input_0, input_2}, std::vector<PatternVar const*>{output_0});
    auto* matmul_1 = builder.AddInstr(
        "matmul", std::vector<PatternVar const*>{input_0, input_1}, std::vector<PatternVar const*>{output_1});
    CHECK_EQ(builder.cur_id(), 6);

    Digraph graph = builder.release();
    CHECK_EQ(graph.nodes().size(), 7u);
    CHECK_EQ(graph.adj().size(), 5u);
    return graph;
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

  Digraph src_pattern = generate_src_pattern();
  Digraph program     = ProgramGraphBuilder(generate_program()).release();
  CHECK_EQ(program.nodes().size(), 7u);
  PatternMatcher matcher(src_pattern, program);
  auto patterns = matcher.DetectPatterns();
  CHECK_EQ(patterns.size(), 1u);
}

}  // namespace cinn::frontend::pass