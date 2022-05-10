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

std::ostream& operator<<(std::ostream& os, const Node& node);

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
  bool Tell(const Node* var) const override;

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

  bool Tell(const Node* instr) const override;

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
  bool operator()(const Node* lhs, const Node* rhs) const;

  template <typename T, typename = std::enable_if_t<std::is_base_of<Node, T>::value>>
  bool operator()(const std::unique_ptr<T>& lhs, const std::unique_ptr<T>& rhs) const {
    CHECK(lhs && rhs);
    return lhs->id() < rhs->id();
  }

  bool operator()(const std::pair<Node const*, Node const*>& lhs, const std::pair<Node const*, Node const*>& rhs) const;
};

class Adjacent {
 public:
  size_t size() const { return adj_.size(); }
  void Add(Node const* start, Node const* end, int16_t idx) { adj_[start].emplace(Target(end, idx)); }
  std::set<std::pair<Node const*, Node const*>, NodeLessThan> edges() const;
  bool HasEdge(Node const* start, Node const* end) const;

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
  PatternVar* AddVar();

  PatternInstr* AddInstr(const char* type,
                         const std::vector<PatternVar const*>& inputs,
                         const std::vector<PatternVar const*>& outputs);

  int16_t cur_id() const { return cur_id_; }
  Digraph release() override { return std::move(graph_); }

 private:
  int16_t cur_id_{-1};
};

class ProgramGraphBuilder final : public GraphBuilder {
 public:
  ProgramGraphBuilder(const Program& program);
  Digraph release() override { return std::move(graph_); }

 private:
  void AddInstr(const _Instruction_* instr);
  bool VarExists(const _Variable_* var) const { return var_map_.count(var); }
  void AddVar(const _Variable_* var);

  int16_t cur_id_{-1};
  std::map<const _Variable_*, ProgramVar*> var_map_;
};

// TODO: use a more classical algorithm.
class PatternMatcher {
 public:
  using pattern_node_t = Node;
  using program_node_t = Node;
  PatternMatcher(const Digraph& pattern, const Digraph& program);
  std::vector<std::map<pattern_node_t const*, program_node_t const*>> DetectPatterns();

 private:
  class HitGroup {
   public:
    const std::map<pattern_node_t const*, program_node_t const*, NodeLessThan>& roles() const { return roles_; }
    void Register(program_node_t const* node, pattern_node_t const* pat) {
      roles_[pat] = node;
      nodes_.insert(node);
    }
    bool Match(program_node_t const* node, pattern_node_t const* pat) const;

   private:
    std::map<pattern_node_t const*, program_node_t const*, NodeLessThan> roles_;
    std::set<program_node_t const*> nodes_;
  };

  void NodeMatch();
  Digraph const* program_{};
  Digraph const* pattern_{};
  std::map<pattern_node_t const*, std::vector<program_node_t const*>, NodeLessThan> pdnodes2nodes_;
  std::set<std::pair<Node const*, Node const*>, NodeLessThan> pattern_edges_;
  std::vector<HitGroup> groups_;
};

}  // namespace cinn::frontend::pass
