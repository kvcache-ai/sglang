import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=1, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=1, suite="stage-b-test-small-1-gpu-amd")


class TestSWAChunkEviction(unittest.TestCase):
    def setUp(self):
        server_args = SimpleNamespace(enable_piecewise_cuda_graph=False)
        patcher = patch(
            "sglang.srt.managers.schedule_batch.get_global_server_args",
            return_value=server_args,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def create_batch(
        self,
        *,
        current_prefix_len: int,
        last_extend_prefix_len: int,
        extend_batch_idx: int,
        enable_overlap: bool = True,
    ):
        req = MagicMock(spec=Req)
        req.extend_batch_idx = extend_batch_idx
        req.last_extend_prefix_len = last_extend_prefix_len

        tree_cache = MagicMock()
        tree_cache.supports_swa.return_value = True
        tree_cache.is_chunk_cache.return_value = True
        tree_cache.sliding_window_size = 4096

        batch = ScheduleBatch(
            reqs=[req],
            tree_cache=tree_cache,
            forward_mode=ForwardMode.EXTEND,
            enable_overlap=enable_overlap,
        )
        batch.prefix_lens = [current_prefix_len]
        return batch, req

    def test_overlap_uses_actual_previous_extend_boundary(self):
        batch, req = self.create_batch(
            current_prefix_len=5632,
            last_extend_prefix_len=4096,
            extend_batch_idx=2,
        )

        with patch.object(batch, "_evict_swa") as evict_swa:
            batch.maybe_evict_swa()

        evict_swa.assert_called_once_with(req, 4096)

    def test_overlap_keeps_fixed_size_chunk_behavior(self):
        batch, req = self.create_batch(
            current_prefix_len=8192,
            last_extend_prefix_len=4096,
            extend_batch_idx=2,
        )

        with patch.object(batch, "_evict_swa") as evict_swa:
            batch.maybe_evict_swa()

        evict_swa.assert_called_once_with(req, 4096)

    def test_overlap_does_not_evict_first_two_extend_batches(self):
        for extend_batch_idx in (0, 1):
            with self.subTest(extend_batch_idx=extend_batch_idx):
                batch, _ = self.create_batch(
                    current_prefix_len=4096,
                    last_extend_prefix_len=0,
                    extend_batch_idx=extend_batch_idx,
                )

                with patch.object(batch, "_evict_swa") as evict_swa:
                    batch.maybe_evict_swa()

                evict_swa.assert_not_called()

    def test_non_overlap_uses_current_prefix_boundary(self):
        batch, req = self.create_batch(
            current_prefix_len=5632,
            last_extend_prefix_len=4096,
            extend_batch_idx=2,
            enable_overlap=False,
        )

        with patch.object(batch, "_evict_swa") as evict_swa:
            batch.maybe_evict_swa()

        evict_swa.assert_called_once_with(req, 5632)

    def test_prepare_for_extend_records_actual_prefix_after_allocation(self):
        req = Req("test", "", [1, 2, 3, 4, 5], SamplingParams())
        req.fill_ids = [1, 2, 3, 4, 5]
        req.prefix_indices = torch.tensor([10, 11], dtype=torch.int64)
        req.set_extend_input_len(3)

        model_config = SimpleNamespace(
            is_matryoshka=False,
            is_encoder_decoder=False,
            vocab_size=128,
        )
        batch = ScheduleBatch(reqs=[req], model_config=model_config, device="cpu")

        allocation = (
            torch.tensor([20, 21, 22], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
            [0],
        )
        server_args = MagicMock()
        server_args.enable_mamba_extra_buffer.return_value = False

        with (
            patch(
                "sglang.srt.managers.schedule_batch.alloc_for_extend",
                return_value=allocation,
            ),
            patch(
                "sglang.srt.managers.schedule_batch.get_global_server_args",
                return_value=server_args,
            ),
            patch(
                "sglang.srt.managers.schedule_batch.SamplingBatchInfo.from_schedule_batch"
            ),
        ):
            batch.prepare_for_extend()

        self.assertEqual(req.last_extend_prefix_len, 2)
        self.assertEqual(req.extend_batch_idx, 1)

    def test_retraction_resets_actual_extend_boundary(self):
        req = Req("test", "", [1], SamplingParams())
        req.last_extend_prefix_len = 1536
        req.extend_batch_idx = 2

        req.reset_for_retract()

        self.assertEqual(req.last_extend_prefix_len, 0)
        self.assertEqual(req.extend_batch_idx, 0)


if __name__ == "__main__":
    unittest.main()
