# SPDX-License-Identifier: Apache-2.0

import unittest

import torch


class TestSharedFullContext(unittest.TestCase):

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
    def test_restore_raw_integer_parameter_disables_grad(self):
        from sglang.srt.layers.moe.kt_ep_wrapper import SharedFullContext

        device = torch.device("cuda:0")
        ctx = object.__new__(SharedFullContext)
        ctx.gpu_layer = torch.nn.Module()
        ctx.gpu_layer.w13_weight = torch.nn.Parameter(
            torch.empty((1,), dtype=torch.float32, device=device)
        )
        ctx._raw_weight_shapes = {
            "w13_weight": ((4, 8), torch.int32, device),
        }

        ctx._restore_raw_attrs()

        restored = ctx.gpu_layer.w13_weight
        self.assertEqual(tuple(restored.shape), (4, 8))
        self.assertEqual(restored.dtype, torch.int32)
        self.assertEqual(restored.device, device)
        self.assertFalse(restored.requires_grad)


if __name__ == "__main__":
    unittest.main()
