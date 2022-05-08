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

struct NodeComp {
  bool operator()(const Node* lhs, const Node* rhs) const {
    CHECK(lhs && rhs);
    return lhs->id() < rhs->id();
  }
  bool operator()(const std::unique_ptr<Node>& lhs, const std::unique_ptr<Node>& rhs) const {
    CHECK(lhs && rhs);
    return lhs->id() < rhs->id();
  }
};

class VarRepr final : public Node {
 public:
  VarRepr* set_external(bool value) {
    external_ = value;
    return this;
  }

 private:
  bool external_{false};
  std::vector<std::function<bool(const Variable&)>> tellers_;
};

class InstrRepr final : public Node {
 public:
  InstrRepr(const char* type, std::vector<VarRepr const*>&& inputs, std::vector<VarRepr const*>&& outputs)
      : inputs_{std::move(inputs)}, outputs_{std::move(outputs)} {
    tellers_.emplace_back([=](const Instruction& instr) -> bool { return instr->op_type == type; });
  }

  const std::vector<VarRepr const*>& inputs() const { return inputs_; }

  const std::vector<VarRepr const*>& outputs() const { return outputs_; }

 private:
  std::vector<std::function<bool(const Instruction&)>> tellers_;
  std::vector<VarRepr const*> inputs_;
  std::vector<VarRepr const*> outputs_;
};

class Pattern {
 public:
  template <typename... Args>
  VarRepr* AddVar(Args&&... args) {
    CHECK(!finished_);
    auto var = std::make_unique<VarRepr>(std::forward<Args>(args)...);
    var->set_id(++cur_id_);
    VarRepr* ret = var.get();
    nodes_.insert(std::move(var));
    return ret;
  }

  template <typename... Args>
  InstrRepr* AddInstr(Args&&... args) {
    CHECK(!finished_);
    auto instr = std::make_unique<InstrRepr>(std::forward<Args>(args)...);
    instr->set_id(++cur_id_);
    InstrRepr* ret = instr.get();
    nodes_.insert(std::move(instr));
    return ret;
  }

  int16_t cur_id() const { return cur_id_; }

  const std::set<std::unique_ptr<Node>, NodeComp>& nodes() const { return nodes_; }

  void Finish() { finished_ = true; }

 private:
  void GenerateVarOuts() {
    for (const auto& node : nodes_) {
      const auto* instr = dynamic_cast<InstrRepr const*>(node.get());
      if (instr) {
        for (const auto* output : instr->outputs()) {
          var_outs_[output].emplace_back(instr);
        }
      }
    }
  }

  int16_t cur_id_{-1};
  bool finished_{false};
  std::set<std::unique_ptr<Node>, NodeComp> nodes_;
  std::map<VarRepr const*, std::vector<InstrRepr const*>, NodeComp> var_outs_;
};

class PatternMatcher {
 public:
 private:
};

}  // namespace cinn::frontend::pass
