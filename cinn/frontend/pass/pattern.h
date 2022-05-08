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
  Node(const Node&) = delete;
  int16_t id() const { return id_; }
  void set_id(int16_t id) { id_ = id; }

 private:
  int16_t id_{-1};
};

bool operator<(const Node& lhs, const Node& rhs) { return lhs.id() < rhs.id(); }

class VarRepr final : public Node {
 public:
 private:
  Variable* var_{};
  std::vector<std::function<bool(const Variable&)>> tellers_;
};

class InstrRepr final : public Node {
 public:
  InstrRepr(const char* type, std::vector<VarRepr*>&& inputs, std::vector<VarRepr*>&& outputs)
      : inputs_{std::move(inputs)}, outputs_{std::move(outputs)} {
    tellers_.emplace_back([=](const Instruction& instr) -> bool { return instr->op_type == type; });
  }

 private:
  Instruction* instr_{};
  std::vector<std::function<bool(const Instruction&)>> tellers_;
  std::vector<VarRepr*> inputs_;
  std::vector<VarRepr*> outputs_;
};

class Pattern {
 public:
  template <typename... Args>
  VarRepr* AddVar(Args&&... args) {
    auto var = std::make_unique<VarRepr>(std::forward<Args>(args)...);
    var->set_id(cur_id_++);
    VarRepr* ret = var.get();
    nodes_.insert(std::move(var));
    return ret;
  }

  template <typename... Args>
  InstrRepr* AddInstr(Args&&... args) {
    auto instr = std::make_unique<InstrRepr>(std::forward<Args>(args)...);
    instr->set_id(cur_id_++);
    InstrRepr* ret = instr.get();
    nodes_.insert(std::move(instr));
    return ret;
  }

 private:
  uint16_t cur_id_{};
  std::set<std::unique_ptr<Node>> nodes_;
};

}  // namespace cinn::frontend::pass
