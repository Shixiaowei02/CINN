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

#include "cinn/hlir/framework/schedule.h"

#include "cinn/common/cinn_value.h"
#include "cinn/hlir/pe/ir_schedule_pe.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir_schedule.h"

DECLARE_bool(cinn_ir_schedule);

namespace cinn {
namespace hlir {
namespace framework {

lang::PackedFunc GetInjectiveScheduleFunc(const std::vector<std::vector<int>> &output_shapes, const Target &target) {
  return lang::PackedFunc([=](lang::Args args, lang::RetValue *ret) {
    if (FLAGS_cinn_ir_schedule) {
      CHECK(!args.empty()) << "The input argument of schedule is empty! Please check.\n";
      common::CINNValuePack arg_pack = args[0];
      CHECK_EQ(arg_pack.size(), 1UL);
      Expr ast_expr = arg_pack[0];
      std::vector<Expr> vec_ast{ast_expr};
      ir::ModuleExpr mod_expr(vec_ast);
      ir::IRSchedule ir_sch(mod_expr);
      if (target.arch == Target::Arch::NVGPU) {
        pe::IRCudaScheduleInjective(ir_sch, output_shapes.front(), target);
      } else if (target.arch == Target::Arch::X86) {
        pe::IRScheduleInjectiveCPU(ir_sch, output_shapes.front(), target);
      }
      std::vector<common::CINNValue> res;
      res.push_back(arg_pack[0]);
      *ret = common::CINNValuePack{res};
    } else {
      CHECK(!args.empty()) << "The input argument of schedule is empty! Please check.\n";
      common::CINNValuePack arg_pack = args[0];
      CHECK_EQ(arg_pack.size(), 2UL);
      if (target.arch == Target::Arch::NVGPU) {
        Expr out              = arg_pack[0];
        poly::StageMap stages = arg_pack[1];
        CHECK(out.as_tensor());
        pe::CudaScheduleInjective(stages[out.as_tensor_ref()], output_shapes.front(), target);
      } else if (target.arch == Target::Arch::X86) {
        Expr out              = arg_pack[0];
        poly::StageMap stages = arg_pack[1];
        CHECK(out.as_tensor());
        pe::ScheduleInjectiveCPU(stages[out.as_tensor_ref()], output_shapes.front(), target);
      }
      *ret = arg_pack;
    }
  });
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
