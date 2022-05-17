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

#include <fstream>
#include <iostream>

#include "cinn/frontend/cinn_builder.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/pass/pattern.h"
#include "cinn/frontend/program_pass.h"
#include "glog/logging.h"

namespace cinn::frontend::pass {

class DotMergerPass : public ProgramPass {
 public:
  DotMergerPass(const std::string& name) : ProgramPass(name) { pattern_ = std::move(GeneratePattern()); }

 protected:
  void ApplyImpl(Program* prog,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) override {
    if (!Match(prog)) {
      return;
    }
    Rewrite(prog, fetch_ids, target);
  }

 private:
  std::unique_ptr<Digraph> GeneratePattern();
  bool Match(Program* prog);

  // TODO: More general rewrite logic.
  void Rewrite(Program* prog, const std::unordered_set<std::string>& fetch_ids, const common::Target& target);

 private:
  std::unique_ptr<Digraph> pattern_;
  std::unique_ptr<Digraph> program_;
  PatternMatcher matcher_;
  std::vector<PatternMatcher::pattern_map_t> matches_;
};

template <typename T>
T GetAttr(const Instruction* instr, const char* attr, T default_value) {
  auto& attrs = instr->get()->attrs;
  if (attrs.count(attr)) {
    return absl::get<T>(attrs.at(attr));
  } else {
    return default_value;
  }
}

std::unique_ptr<Digraph> DotMergerPass::GeneratePattern() {
  auto has_2d_shape = [](ProgramVar* var) -> bool { return var->raw()->get()->shape.size() == 2; };

  // TODO: move it into the base class
  auto in_matmul = [](ProgramVar* var) -> bool {
    for (auto target : var->prog()->adj().GetTargets(var)) {
      auto* prog = dynamic_cast<ProgramInstr*>(target.end());
      if (prog && prog->raw()->get()->op_type == "matmul" && !GetAttr(prog->raw(), "trans_a", false) &&
          !GetAttr(prog->raw(), "trans_b", false)) {
        return true;
      }
    }
    return false;
  };
  auto out_matmul = [](ProgramVar* var) -> bool {
    bool res = false;
    for (auto edge : var->prog()->adj().edges()) {
      auto* first = dynamic_cast<ProgramInstr*>(edge.first);
      if (first && first->raw()->get()->op_type == "matmul" && edge.second == var) {
        return true;
      }
    }
    return false;
  };
  auto no_trans = [](ProgramInstr* instr) -> bool {
    bool res = false;
    return !GetAttr(instr->raw(), "trans_a", false) && !GetAttr(instr->raw(), "trans_b", false);
  };

  PatternBuilder builder;
  auto* in_0     = builder.AddVar()->Assert(has_2d_shape)->Assert(in_matmul)->set_label("in_0");
  auto* in_1     = builder.AddVar()->Assert(has_2d_shape)->Assert(in_matmul)->set_label("in_1");
  auto* in_2     = builder.AddVar()->Assert(has_2d_shape)->Assert(in_matmul)->set_label("in_2");
  auto* out_0    = builder.AddVar()->Assert(has_2d_shape)->Assert(out_matmul)->set_label("out_0");
  auto* out_1    = builder.AddVar()->Assert(has_2d_shape)->Assert(out_matmul)->set_label("out_1");
  auto* matmul_0 = builder.AddInstr("matmul", std::vector<PatternVar*>{in_0, in_1}, std::vector<PatternVar*>{out_0})
                       ->set_label("matmul_0")
                       ->Assert(no_trans);
  auto* matmul_1 = builder.AddInstr("matmul", std::vector<PatternVar*>{in_0, in_2}, std::vector<PatternVar*>{out_1})
                       ->set_label("matmul_1")
                       ->Assert(no_trans);
  return builder.release();
};

bool DotMergerPass::Match(Program* prog) {
  program_ = std::move(ProgramGraphBuilder(*prog).release());
  PatternMatcher matcher;
  matcher.Init(*pattern_, *program_);
  matches_ = std::move(matcher.DetectPatterns());
  VLOG(5) << "matches size " << matches_.size();
  return matches_.size();
}

void print_shape(const std::string& str, const Variable& var) {
  std::cout << str << ' ';
  for (int i : var->shape) {
    std::cout << i << ", ";
  }
  std::cout << '\n';
}

bool print_matmul(const Instruction& matmul_instr) {
  auto& attrs = matmul_instr->attrs;
  bool trans_a{}, trans_b{}, trans_out{};
  if (attrs.count("trans_a")) {
    trans_a = absl::get<bool>(attrs.at("trans_a"));
  }
  if (attrs.count("trans_b")) {
    trans_b = absl::get<bool>(attrs.at("trans_b"));
  }
  if (attrs.count("trans_out")) {
    trans_out = absl::get<bool>(attrs.at("trans_out"));
  }
  std::cout << trans_a << ", " << trans_b << ", " << trans_out << '\n';
  if (!trans_a && !trans_b) {
    return true;
  }
  return false;
}

int input_idx(const Instruction& instr, const Variable& var) {
  int res = -1;
  for (size_t i = 0; i < instr->inputs.size(); ++i) {
    if (instr->inputs[i].get() == var.get()) {
      res = i;
    }
  }
  return res;
}

// update the interfaces of the graph to use priority traversal.
void union_find(std::vector<std::pair<Instruction, bool>>& instrs, const _Instruction_* instr) {
  std::set<_Variable_*> vars{};
  for (auto& input : instr->inputs) {
    vars.emplace(input.get());
  }
  while (true) {
    uint64_t before_size = vars.size();
    for (auto& instr : instrs) {
      for (auto& output : instr.first->outputs) {
        if (vars.count(output.get()) != 0) {
          // LOG(INFO) << "find: " << instr.first->op_type;
          instr.second = true;
          break;
        }
        // LOG(INFO) << "not-find: " << instr.first->op_type;
      }
      if (instr.second) {
        for (auto& input : instr.first->inputs) {
          vars.emplace(input.get());
        }
      }
    }
    if (vars.size() == before_size) {
      break;
    }
  }
}

void DotMergerPass::Rewrite(Program* prog,
                            const std::unordered_set<std::string>& fetch_ids,
                            const common::Target& target) {
  {
    std::stringstream ss;
    ss << *prog;
    std::string myString = ss.str();

    std::ofstream file("program_before.txt", std::ofstream::out | std::ofstream::trunc);
    file << myString;
  }

  for (const auto& match : matches_) {
    const Variable& in_0  = *GetMatchedVar(match, "in_0");
    const Variable& in_1  = *GetMatchedVar(match, "in_1");
    const Variable& in_2  = *GetMatchedVar(match, "in_2");
    const Variable& out_0 = *GetMatchedVar(match, "out_0");
    const Variable& out_1 = *GetMatchedVar(match, "out_1");

    const auto& matmul_0 = *GetMatchedInstr(match, "matmul_0");
    const auto& matmul_1 = *GetMatchedInstr(match, "matmul_1");

    int idx00 = input_idx(matmul_0, in_0);
    int idx01 = input_idx(matmul_1, in_0);
    int idx1  = input_idx(matmul_0, in_1);
    int idx2  = input_idx(matmul_1, in_2);

    bool lhs{true};
    int axis{1};
    if (idx00 != idx01 || idx1 != idx2) {
      continue;
    } else if (idx00 == 1) {
      lhs  = false;
      axis = 0;
    }

    std::cout << "[[" << idx00 << ", " << idx01 << ", " << idx1 << ", " << idx2 << " ]]\n";

    std::cout << "axis = " << axis << '\n';
    print_shape("in_0", in_0);
    print_shape("in_1", in_1);
    print_shape("in_2", in_2);
    print_shape("out_0", out_0);
    print_shape("out_1", out_1);

    std::set<_Instruction_*> nodes_to_remove{GetMatchedInstr(match, "matmul_0")->get(),
                                             GetMatchedInstr(match, "matmul_1")->get()};

    NetBuilder builder("dot_merger_builder");
    Variable slice0_out;
    Variable slice1_out;
    bool rewrited{false};

    auto insert_pattern = [&]() {
      rewrited = true;
      Variable matmul_out;
      auto concat_out = builder.Concat({in_1, in_2}, axis);
      if (!lhs) {
        matmul_out = builder.Matmul(concat_out, in_0);
      } else {
        matmul_out = builder.Matmul(in_0, concat_out);
      }
      std::cout << "in_1->shape[axis] = " << in_1->shape[axis] << ", " << in_2->shape[axis] << '\n';
      slice0_out = builder.Slice(matmul_out, {axis}, {0}, {in_1->shape[axis]});
      slice1_out = builder.Slice(matmul_out, {axis}, {in_1->shape[axis]}, {in_1->shape[axis] + in_2->shape[axis]});
    };

    int first_loc{-1};
    std::vector<std::pair<Instruction, bool>> interval;
    for (size_t i = 0; i < prog->size(); ++i) {
      auto& instr = (*prog)[i];
      auto it     = nodes_to_remove.find(instr.get());
      if (it != nodes_to_remove.end()) {
        nodes_to_remove.erase(it);
        if (first_loc == -1) {
          first_loc = i;
        } else if (i == first_loc + 1) {
          LOG(INFO) << "here!";
          insert_pattern();
        } else {
          LOG(INFO) << "not!!! " << i - first_loc;
        }
      }
    }
    if (rewrited) {
      for (size_t i = 0; i < prog->size(); ++i) {
        auto& instr = (*prog)[i];
        for (auto& var : instr->inputs) {
          if (var.get() == out_0.get()) {
            var = slice0_out;
            CHECK(var.get() == slice0_out.get());
          }
          if (var.get() == out_1.get()) {
            var = slice1_out;
            CHECK(var.get() == slice1_out.get());
          }
        }
      }
      auto program = builder.Build();
      *prog        = std::move(program);
    }

    // break;
  }
  LOG(INFO) << "program!!";

  {
    std::stringstream ss;
    ss << *prog;
    std::string myString = ss.str();

    std::ofstream file("program.txt", std::ofstream::out | std::ofstream::trunc);
    file << myString;
  }
  LOG(INFO) << "end of pass";
}

}  // namespace cinn::frontend::pass

CINN_REGISTER_HELPER(DotMerger) {
  CINN_REGISTER_PROGRAM_PASS(DotMerger, ::cinn::frontend::pass::DotMergerPass);
  return true;
}
