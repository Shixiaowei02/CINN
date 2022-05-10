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

#include <array>
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

  virtual bool Tell(const Node* instr) const { return false; }

  int16_t id() const { return id_; }
  void set_id(int16_t id) { id_ = id; }

 private:
  int16_t id_{-1};
};

std::ostream& operator<<(std::ostream& os, const Node& node) {
  os << "[" << &node << "] Node id : " << node.id();
  return os;
}

class ProgramVar final : public Node {
 public:
  ProgramVar(const _Variable_* var) : var_{var} {}
  const _Variable_* raw() const { return var_; }

 private:
  const _Variable_* var_{};
};

class ProgramInstr final : public Node {
 public:
  ProgramInstr(const _Instruction_* instr) : instr_{instr} {}
  const _Instruction_* raw() const { return instr_; }

 private:
  const _Instruction_* instr_{};
};

class PatternVar final : public Node {
 public:
  bool Tell(const Node* var) const override {
    bool ret          = true;
    const auto* p_var = dynamic_cast<ProgramVar const*>(var);
    if (p_var) {
      for (const auto& teller : tellers_) {
        ret = ret && teller(p_var->raw());
      }
    } else {
      ret = false;
    }
    return ret;
  }

 private:
  bool external_{false};
  std::vector<std::function<bool(const _Variable_*)>> tellers_;
};

class PatternInstr final : public Node {
 public:
  PatternInstr(const char* type) : type_{type} {
    tellers_.emplace_back([=](const _Instruction_* instr) -> bool { return instr->op_type == type_; });
  }
  const char* type() const { return type_; }

  bool Tell(const Node* instr) const override {
    bool ret            = true;
    const auto* p_instr = dynamic_cast<ProgramInstr const*>(instr);
    if (p_instr) {
      for (const auto& teller : tellers_) {
        ret = ret && teller(p_instr->raw());
      }
    } else {
      ret = false;
    }
    return ret;
  }

 private:
  const char* type_{};
  std::vector<std::function<bool(const _Instruction_*)>> tellers_;
};

class Target {
 public:
  Target(const Node* end, int16_t idx) : end_{end}, var_idx_{idx} {}
  explicit Target(const Node* end) : end_{end} {}
  const Node* end() const { return end_; }
  int16_t var_idx() const { return var_idx_; }

 private:
  const Node* end_{};
  int16_t var_idx_{-1};
};

bool operator<(const Target& lhs, const Target& rhs) { return lhs.end() < rhs.end(); }

struct NodeLessThan {
  bool operator()(const Node* lhs, const Node* rhs) const {
    CHECK(lhs && rhs);
    return lhs->id() < rhs->id();
  }
  template <typename T, typename = std::enable_if_t<std::is_base_of<Node, T>::value>>
  bool operator()(const std::unique_ptr<T>& lhs, const std::unique_ptr<T>& rhs) const {
    CHECK(lhs && rhs);
    return lhs->id() < rhs->id();
  }
  bool operator()(const std::pair<Node const*, Node const*>& lhs,
                  const std::pair<Node const*, Node const*>& rhs) const {
    bool res = false;
    if (lhs.first->id() < rhs.first->id()) {
      res = true;
    } else if (lhs.first->id() == rhs.first->id()) {
      res = lhs.second->id() < rhs.second->id();
    }
    return res;
  }
};

class Adjacent {
 public:
  size_t size() const { return adj_.size(); }
  void Add(Node const* start, Node const* end, int16_t idx) { adj_[start].emplace(Target(end, idx)); }
  std::set<std::pair<Node const*, Node const*>, NodeLessThan> edges() const {
    std::set<std::pair<Node const*, Node const*>, NodeLessThan> ret;
    for (const auto& pair : adj_) {
      for (const auto& target : pair.second) {
        ret.emplace(std::pair<Node const*, Node const*>(pair.first, target.end()));
      }
    }
    return ret;
  }
  bool HasEdge(Node const* start, Node const* end) const {
    if (!adj_.count(start)) {
      return false;
    }
    return adj_.at(start).count(Target(end, 0));
  }

 private:
  std::map<Node const*, std::set<Target>, NodeLessThan> adj_;
};

class Digraph {
 public:
  Digraph(const Digraph&) = delete;
  Digraph(Digraph&&)      = default;

  Node* AddNode(std::unique_ptr<Node>&& node) {
    auto* ret = nodes_.emplace(std::move(node)).first->get();
    return ret;
  }
  template <typename... Args>
  void AddEdge(Args... args) {
    adj_.Add(std::forward<Args>(args)...);
  }
  const std::set<std::unique_ptr<Node>, NodeLessThan>& nodes() const { return nodes_; }
  const Adjacent& adj() const { return adj_; }

  // TODO: check for directed acyclic.
 private:
  Digraph() = default;
  std::set<std::unique_ptr<Node>, NodeLessThan> nodes_;
  Adjacent adj_;

  friend class GraphBuilder;
};

class GraphBuilder {
 public:
  GraphBuilder()                    = default;
  GraphBuilder(const GraphBuilder&) = delete;
  GraphBuilder(GraphBuilder&&)      = default;
  virtual ~GraphBuilder()           = default;
  virtual Digraph release()         = 0;

 protected:
  Digraph graph_;
};

class PatternBuilder final : public GraphBuilder {
 public:
  PatternVar* AddVar() {
    auto var = std::make_unique<PatternVar>();
    var->set_id(++cur_id_);
    PatternVar* ret = var.get();
    graph_.AddNode(std::move(var));
    return ret;
  }

  PatternInstr* AddInstr(const char* type,
                         const std::vector<PatternVar const*>& inputs,
                         const std::vector<PatternVar const*>& outputs) {
    auto instr = std::make_unique<PatternInstr>(type);
    instr->set_id(++cur_id_);
    PatternInstr* ret = instr.get();
    graph_.AddNode(std::move(instr));

    for (size_t i = 0; i < inputs.size(); ++i) {
      graph_.AddEdge(inputs[i], ret, i);
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      graph_.AddEdge(ret, outputs[i], i);
    }
    return ret;
  }

  int16_t cur_id() const { return cur_id_; }
  Digraph release() override { return std::move(graph_); }

 private:
  int16_t cur_id_{-1};
};

class ProgramGraphBuilder final : public GraphBuilder {
 public:
  ProgramGraphBuilder(const Program& program) {
    for (size_t i = 0; i < program.size(); ++i) {
      AddInstr(program[i].get());
    }
  }
  Digraph release() override { return std::move(graph_); }

 private:
  void AddInstr(const _Instruction_* instr) {
    auto p_instr    = std::make_unique<ProgramInstr>(instr);
    auto* raw_instr = p_instr.get();
    p_instr->set_id(++cur_id_);
    graph_.AddNode(std::move(p_instr));

    for (size_t i = 0; i < instr->inputs.size(); ++i) {
      auto* raw_var = instr->inputs[i].get();
      if (!VarExists(raw_var)) {
        AddVar(raw_var);
      }
      graph_.AddEdge(var_map_[raw_var], raw_instr, i);
    }
    for (size_t i = 0; i < instr->outputs.size(); ++i) {
      auto* raw_var = instr->outputs[i].get();
      if (!VarExists(raw_var)) {
        AddVar(raw_var);
      }
      graph_.AddEdge(raw_instr, var_map_[raw_var], i);
    }
  }

  bool VarExists(const _Variable_* var) const { return var_map_.count(var); }

  void AddVar(const _Variable_* var) {
    CHECK(!VarExists(var)) << "Repeated addition of variables is not allowed.";
    auto p_var = std::make_unique<ProgramVar>(var);
    auto* raw  = p_var.get();
    p_var->set_id(++cur_id_);
    graph_.AddNode(std::move(p_var));
    var_map_[var] = raw;
  }

  int16_t cur_id_{-1};
  std::map<const _Variable_*, ProgramVar*> var_map_;
};

// TODO: use a more classical algorithm.
class PatternMatcher {
 public:
  using pattern_node_t = Node;
  using program_node_t = Node;
  PatternMatcher(const Digraph& pattern, const Digraph& program) : program_{&program}, pattern_{&pattern} {
    pattern_edges_ = pattern_->adj().edges();
    NodeMatch();
    VLOG(5) << "[Program Edge]";
    for (auto& a : program_->adj().edges()) {
      VLOG(5) << *(a.first) << " -> " << *(a.second);
    }
    VLOG(5) << "[Pattern Edge]";
    for (auto& a : pattern_->adj().edges()) {
      VLOG(5) << *(a.first) << " -> " << *(a.second);
    }
  }

  std::vector<std::map<pattern_node_t const*, program_node_t const*>> DetectPatterns() {
    std::vector<std::map<pattern_node_t const*, program_node_t const*>> res;
    std::array<std::vector<HitGroup>, 2> bi_records;
    auto& init_groups = bi_records[0];

    auto* first_pnode = pdnodes2nodes_.begin()->first;
    if (!pdnodes2nodes_.count(first_pnode)) {
      return res;
    }
    for (auto* node : pdnodes2nodes_[first_pnode]) {
      HitGroup group;
      group.Register(node, first_pnode);
      init_groups.emplace_back(std::move(group));
    }
    int step{0};
    for (const auto& edge : pattern_edges_) {
      auto& pre_groups = bi_records[step % 2];
      auto& cur_groups = bi_records[1 - (step++ % 2)];
      cur_groups.clear();
      for (const auto* source : pdnodes2nodes_[edge.first]) {
        for (const auto* target : pdnodes2nodes_[edge.second]) {
          for (const auto& group : pre_groups) {
            if (program_->adj().HasEdge(source, target)) {
              HitGroup new_group = group;
              bool flag          = new_group.Match(source, edge.first) && new_group.Match(target, edge.second);
              if (flag) {
                new_group.Register(source, edge.first);
                new_group.Register(target, edge.second);
                cur_groups.push_back(new_group);
              }
            }
          }
        }
      }
    }

    // TODO: Distinguishing and processing of external nodes.
    std::set<program_node_t const*> visited;
    for (auto& group : bi_records[step % 2]) {
      std::map<pattern_node_t const*, program_node_t const*> subgraph;
      bool overlapped{false};
      for (auto& role : group.roles()) {
        if (visited.find(role.second) == visited.end()) {
          subgraph.emplace(role.first, role.second);
        } else {
          overlapped = true;
        }
      }
      if (!overlapped) {
        for (auto& role : group.roles()) {
          visited.emplace(role.second);
        }
        VLOG(5) << "[Matched] : pattern -> program";
        for (auto& pair : subgraph) {
          VLOG(5) << "   -- " << *(pair.first) << " -> " << *(pair.second);
        }
        res.emplace_back(std::move(subgraph));
      }
    }
    return res;
  }

 private:
  class HitGroup {
   public:
    const std::map<pattern_node_t const*, program_node_t const*, NodeLessThan>& roles() const { return roles_; }
    void Register(program_node_t const* node, pattern_node_t const* pat) {
      roles_[pat] = node;
      nodes_.insert(node);
    }

    bool Match(program_node_t const* node, pattern_node_t const* pat) const {
      if (nodes_.count(node)) {
        if (roles_.count(pat) && roles_.at(pat) == node) return true;
        return false;
      } else {
        if (roles_.count(pat) && roles_.at(pat) != node) return false;
        return true;
      }
    }

   private:
    std::map<pattern_node_t const*, program_node_t const*, NodeLessThan> roles_;
    std::set<program_node_t const*> nodes_;
  };

  void NodeMatch() {
    for (auto& pt_node : pattern_->nodes()) {
      for (auto& pr_node : program_->nodes()) {
        if (pt_node->Tell(pr_node.get())) {
          pdnodes2nodes_[pt_node.get()].emplace_back(pr_node.get());
        }
      }
    }
  }

  Digraph const* program_{};
  Digraph const* pattern_{};
  std::map<pattern_node_t const*, std::vector<program_node_t const*>, NodeLessThan> pdnodes2nodes_;
  std::set<std::pair<Node const*, Node const*>, NodeLessThan> pattern_edges_;
  std::vector<HitGroup> groups_;
};

}  // namespace cinn::frontend::pass
