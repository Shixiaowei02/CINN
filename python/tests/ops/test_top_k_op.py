#!/usr/bin/env python3

# Copyright (c) 2022 CINN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
from op_test import OpTest, OpTestTool
import paddle
import paddle.nn.functional as F
import cinn
from cinn.frontend import *
from cinn.common import *


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestTopKOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x": np.resize(np.arange(100, dtype=np.float32), [10, 10])
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        out = paddle.topk(x, 5)

        self.paddle_outputs = [out[0], out[1]]

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("top_k")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        out = builder.top_k(x, 5)
        prog = builder.build()
        forward_res = self.get_cinn_output(
            prog, target, [x], [self.inputs["x"]], [out[0], out[1]])

        self.cinn_outputs = forward_res

    def test_check_results(self):
        self.build_paddle_program(self.target)
        self.build_cinn_program(self.target)
        print(self.paddle_outputs[0])
        print(self.paddle_outputs[1])
        print(self.cinn_outputs[0])
        print(self.cinn_outputs[1])
        self.check_results(self.paddle_outputs, self.cinn_outputs, 1e-5, False,
                           False)


if __name__ == "__main__":
    unittest.main()
