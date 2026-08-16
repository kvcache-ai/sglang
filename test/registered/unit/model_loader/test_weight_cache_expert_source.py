# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.srt.weight_cache.expert_source import (
    ExpertSlotKey,
    ExpertSourceDirectory,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestExpertSourceDirectory(CustomTestCase):
    def test_uses_routed_tensor_slot_views_and_whole_tensor_regions(self):
        first = torch.empty(3, 2, 4, dtype=torch.float32)
        second = torch.empty(3, 5, dtype=torch.float16)
        directory = ExpertSourceDirectory.from_routed_experts({7: [first, second]})
        locations = dict(directory.items())

        self.assertEqual(
            directory.pointer_for(locations[ExpertSlotKey(7, 1, 0)]),
            first[1].data_ptr(),
        )
        self.assertEqual(
            locations[ExpertSlotKey(7, 2, 1)].byte_size,
            second[2].numel() * second.element_size(),
        )
        self.assertEqual(len(directory.memory_regions), 2)

    def test_rejects_noncontiguous_routed_expert_layout(self):
        with self.assertRaisesRegex(ValueError, "contiguous"):
            ExpertSourceDirectory.from_routed_experts(
                {0: [torch.empty(2, 3, 4).transpose(1, 2)]}
            )

    def test_deduplicates_regions_without_merging_slot_identity(self):
        tensor = torch.empty(2, 4)
        directory = ExpertSourceDirectory.from_routed_experts({0: [tensor, tensor]})
        locations = dict(directory.items())

        self.assertEqual(len(directory.memory_regions), 1)
        self.assertEqual(
            directory.pointer_for(locations[ExpertSlotKey(0, 1, 0)]),
            directory.pointer_for(locations[ExpertSlotKey(0, 1, 1)]),
        )


if __name__ == "__main__":
    unittest.main()
