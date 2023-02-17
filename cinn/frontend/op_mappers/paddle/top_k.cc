// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void TopKOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();
  CHECK_EQ(op_desc.Output("Indices").size(), 1UL);
  auto indices_name = op_desc.Output("Indices").front();
  auto x            = ctx.GetVar(x_name);

  CHECK(op_desc.HasAttr("k"));
  auto k = utils::GetAttrOrDefault<int>(op_desc, "k");

  auto sort_tmp    = ctx.Builder()->Sort(x, -1, false);
  auto sort_out    = ctx.Builder()->Slice(sort_tmp, {-1}, {0}, {k});
  auto argsort_tmp = ctx.Builder()->ArgSort(x, -1, false);
  auto argsort_out = ctx.Builder()->Slice(argsort_tmp, {-1}, {0}, {k});

  ctx.AddVar(out_name, sort_out);
  ctx.AddVarModelToProgram(out_name, sort_out->id);
  ctx.AddVar(indices_name, argsort_out);
  ctx.AddVarModelToProgram(indices_name, argsort_out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_top_k) {
  CINN_REGISTER_OP_MAPPER(topk, cinn::frontend::paddle_mappers::TopKOpMapper)
  return true;
}
