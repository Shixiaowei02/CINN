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

#include "cinn/ir/ir_schedule.h"

#include "cinn/lang/compute.h"
#include "cinn/lang/lower.h"
#include "gtest/gtest.h"

namespace cinn {
namespace ir {

TEST(IrSchedule, GetExprs) {
  Context::Global().ResetNameId();
  Expr M(32);
  lang::Placeholder<float> A("A", {M});
  auto B = lang::Compute(
      {M}, [&](Var i) { return A(i) * ir::Expr(2.f); }, "B");
  auto C = lang::Compute(
      {M}, [&](Var i) { return B(i) * ir::Expr(1.f); }, "C");
  ir::ModuleExpr mod_expr({A, B, C});
  CHECK_EQ(mod_expr.GetExprs().size(), 3);
  for (const auto& expr : mod_expr.GetExprs()) {
    LOG(INFO) << expr;
    auto* tensor = expr.as_tensor();
    // The placeholder tensor has no body.
    if (tensor && tensor->is_compute_node()) {
      auto body = tensor->body();
      CHECK(tensor->body().defined());
      LOG(INFO) << tensor->body();
    }
  }
  /*
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops("B");
  LOG(INFO) << "loops size = " << loops.size();
  */
}

TEST(IrSchedule, IrBlock) {
  Context::Global().ResetNameId();
  Expr M(32);
  lang::Placeholder<float> A("A", {M});
  auto B = lang::Compute(
      {M}, [&](Var i) { return A(i) * ir::Expr(2.f); }, "B");
  auto C = lang::Compute(
      {M}, [&](Var i) { return B(i) * ir::Expr(1.f); }, "C");
  ir::ModuleExpr mod_expr({ir::Block::Make({A, B, C})});
  CHECK_EQ(mod_expr.GetExprs().size(), 1);
  for (const auto& expr : mod_expr.GetExprs()) {
    LOG(INFO) << expr;
  }
  ir::IRSchedule ir_sch(mod_expr);
  ir_sch.MergeExprs();
  for (const auto& expr : mod_expr.GetExprs()) {
    LOG(INFO) << expr;
  }
}

TEST(IrSchedule, GetLoops) {
  Context::Global().ResetNameId();
  Expr M(32), N(32);
  lang::Placeholder<float> A("A", {M, N});
  auto B = lang::Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");
  /*
  ir::ModuleExpr mod_expr({A, B});
  ir::IRSchedule ir_sch(mod_expr);
  */

  Target target = common::DefaultHostTarget();
  auto stages   = poly::CreateStages({A, B});
  auto func     = cinn::lang::LowerVec("test_split_and_fuse1", stages, {A, B}, {}, {}, nullptr, target, true);
  auto ast_expr = func[0]->body;
  LOG(INFO) << "ast_expr: " << ast_expr;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto fused   = ir_sch.Fuse("B", {0, 1});
  auto splited = ir_sch.Split(fused, {4, -1});
  auto loops   = ir_sch.GetLoops("B");
  LOG(INFO) << "loops size = " << loops.size();
}

}  // namespace ir
}  // namespace cinn
