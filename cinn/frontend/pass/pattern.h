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

#include <memory>
#include <set>

#include "cinn/common/shared.h"
#include "cinn/frontend/syntax.h"

namespace cinn::frontend::pass {

class Node {
 public:
  Node()            = default;
  virtual ~Node()   = default;
  Node(const Node&) = delete;
  int16_t id() const { return id_; }
  void set_id(int16_t id) { id_ = id; }

 private:
  int16_t id_{-1};
};

class VarRepr final : public Node {
 public:
  VarRepr* set_external(bool value) {
    external_ = value;
    return this;
  }

  bool Tell(const _Variable_* var) const {
    bool ret = true;
    for (const auto& teller : tellers_) {
      ret = ret && teller(var);
    }
    return ret;
  }

 private:
  bool external_{false};
  std::vector<std::function<bool(const _Variable_*)>> tellers_;
};

class InstrRepr final : public Node {
 public:
  InstrRepr(const char* type, std::vector<VarRepr const*>&& inputs, std::vector<VarRepr const*>&& outputs)
      : inputs_{std::move(inputs)}, outputs_{std::move(outputs)} {
    tellers_.emplace_back([=](const _Instruction_* instr) -> bool { return instr->op_type == type; });
    tellers_.emplace_back([=](const _Instruction_* instr) -> bool {
      return instr->inputs.size() == inputs_.size() && instr->outputs.size() == outputs_.size();
    });
  }
  const std::vector<VarRepr const*>& inputs() const { return inputs_; }
  const std::vector<VarRepr const*>& outputs() const { return outputs_; }

  bool Tell(const _Instruction_* instr) const {
    bool ret = true;
    for (const auto& teller : tellers_) {
      ret = ret && teller(instr);
    }
    return ret;
  }

 private:
  std::vector<std::function<bool(const _Instruction_*)>> tellers_;
  std::vector<VarRepr const*> inputs_;
  std::vector<VarRepr const*> outputs_;
};

struct NodeComp {
  bool operator()(const Node* lhs, const Node* rhs) const {
    CHECK(lhs && rhs);
    return lhs->id() < rhs->id();
  }
  template <typename T, typename = std::enable_if_t<std::is_base_of<Node, T>::value>>
  bool operator()(const std::unique_ptr<T>& lhs, const std::unique_ptr<T>& rhs) const {
    CHECK(lhs && rhs);
    return lhs->id() < rhs->id();
  }
};

class Pattern {
 public:
  template <typename... Args>
  VarRepr* AddVar(Args&&... args) {
    CheckFinished();
    auto var = std::make_unique<VarRepr>(std::forward<Args>(args)...);
    var->set_id(++cur_id_);
    VarRepr* ret = var.get();
    vars_.insert(std::move(var));
    return ret;
  }

  template <typename... Args>
  InstrRepr* AddInstr(Args&&... args) {
    CheckFinished();
    auto instr = std::make_unique<InstrRepr>(std::forward<Args>(args)...);
    instr->set_id(++cur_id_);
    InstrRepr* ret = instr.get();
    instrs_.insert(std::move(instr));
    return ret;
  }

  int16_t cur_id() const { return cur_id_; }
  const std::map<VarRepr const*, std::vector<InstrRepr const*>, NodeComp>& var_outs() const { return var_repr_outs_; }
  const std::set<std::unique_ptr<VarRepr>, NodeComp>& vars() const { return vars_; }
  const std::set<std::unique_ptr<InstrRepr>, NodeComp>& instrs() const { return instrs_; }
  void Finish() { finished_ = true; }

 private:
  void CheckFinished() const { CHECK(!finished_); }
  void GenerateVarOuts() {
    CheckFinished();
    for (const auto& node : instrs_) {
      const auto* instr = dynamic_cast<InstrRepr const*>(node.get());
      if (instr) {
        for (const auto* output : instr->outputs()) {
          var_repr_outs_[output].emplace_back(instr);
        }
      }
    }
  }

  int16_t cur_id_{-1};
  bool finished_{false};
  std::set<std::unique_ptr<VarRepr>, NodeComp> vars_;
  std::set<std::unique_ptr<InstrRepr>, NodeComp> instrs_;
  std::map<VarRepr const*, std::vector<InstrRepr const*>, NodeComp> var_repr_outs_;
};

class PatternMatcher {
 public:
  PatternMatcher(const Program& program, const Pattern& pattern) : program_{&program}, pattern_{&pattern} {
    for (size_t i = 0; i < program.size(); ++i) {
      const auto& instr = program[i];
      for (const auto& var : instr->inputs) {
        var_outs_[var.get()].emplace_back(instr.get());
      }
    }
    GenerateHitGroup();
  }

  void GenerateHitGroup() {
    for (auto& pt_var : pattern_->vars()) {
      hits_.var_hits.emplace(std::make_pair<VarRepr*, std::vector<_Variable_*>>(pt_var.get(), {}));
      for (auto& var : program_->GetInputs()) {
        if (pt_var->Tell(var.get())) {
          hits_.var_hits[pt_var.get()].emplace_back(var.get());
        }
      }
    }
    for (auto& pt_instr : pattern_->instrs()) {
      hits_.instr_hits.emplace(std::make_pair<InstrRepr*, std::vector<_Instruction_*>>(pt_instr.get(), {}));
      for (size_t i = 0; i < program_->size(); ++i) {
        auto* instr = program_->operator[](i).get();
        if (pt_instr->Tell(instr)) {
          hits_.instr_hits[pt_instr.get()].emplace_back(instr);
        }
      }
    }
  }

  struct Match {
    std::set<std::map<InstrRepr*, _Instruction_*, NodeComp>> pair;
  };

 private:
  struct HitGroup {
    std::map<VarRepr*, std::vector<_Variable_*>, NodeComp> var_hits;
    std::map<InstrRepr*, std::vector<_Instruction_*>, NodeComp> instr_hits;
  };

  Program const* program_{};
  Pattern const* pattern_{};
  // TODO: sequential stability
  std::map<_Variable_ const*, std::vector<_Instruction_ const*>> var_outs_;
  HitGroup hits_;
};

}  // namespace cinn::frontend::pass
