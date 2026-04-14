import argparse
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.checkpoint_engine import checkpoint_engine_worker as worker_module
from sglang.srt.checkpoint_engine import update as update_module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")


class DummyWorker(worker_module.SGLangCheckpointEngineWorkerExtension):
    def get_device_uuid(self) -> str:
        return "GPU-ABC"

    def get_device_id(self) -> int:
        return 3

    def get_model_loader(self):
        return "loader"

    def get_post_hook(self):
        return "hook"


class TestCheckpointEngineUpdate(CustomTestCase):
    def test_build_torchrun_command_uses_module_entrypoint(self):
        command = update_module.build_torchrun_command(
            ["--checkpoint-path", "/tmp/model", "--inference-parallel-size", "4"]
        )

        self.assertEqual(
            command,
            [
                "torchrun",
                "--nproc-per-node=4",
                "-m",
                "sglang.srt.checkpoint_engine.update",
                "--checkpoint-path",
                "/tmp/model",
                "--inference-parallel-size",
                "4",
            ],
        )

    def test_validate_args_requires_exactly_one_input_source(self):
        with self.assertRaisesRegex(ValueError, "must be provided"):
            update_module.validate_args(
                argparse.Namespace(
                    checkpoint_path=None,
                    load_metas_file=None,
                    inference_parallel_size=2,
                )
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            metas_file = os.path.join(tmpdir, "metas.pkl")
            with open(metas_file, "wb"):
                pass

            with self.assertRaisesRegex(ValueError, "cannot be provided together"):
                update_module.validate_args(
                    argparse.Namespace(
                        checkpoint_path=tmpdir,
                        load_metas_file=metas_file,
                        inference_parallel_size=2,
                    )
                )

    def test_validate_args_checks_paths_and_world_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                checkpoint_path=tmpdir,
                load_metas_file=None,
                inference_parallel_size=3,
            )
            with self.assertRaisesRegex(ValueError, "WORLD_SIZE must be divisible"):
                update_module.validate_args(args, world_size=4)

        with self.assertRaisesRegex(ValueError, "Checkpoint path does not exist"):
            update_module.validate_args(
                argparse.Namespace(
                    checkpoint_path="/tmp/does-not-exist",
                    load_metas_file=None,
                    inference_parallel_size=1,
                )
            )

    def test_split_checkpoint_files_is_stable_and_sorted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["b.safetensors", "a.safetensors", "ignore.bin"]:
                with open(os.path.join(tmpdir, name), "wb"):
                    pass

            rank0 = update_module.split_checkpoint_files(tmpdir, rank=0, world_size=2)
            rank1 = update_module.split_checkpoint_files(tmpdir, rank=1, world_size=2)

            self.assertEqual(rank0, [os.path.join(tmpdir, "a.safetensors")])
            self.assertEqual(rank1, [os.path.join(tmpdir, "b.safetensors")])


class TestCheckpointEngineWorker(CustomTestCase):
    def test_update_weights_from_ipc_uses_resolved_handle(self):
        worker = DummyWorker()
        mock_update = MagicMock()

        with patch.object(worker_module, "ce_update_weights_from_ipc", mock_update):
            worker.update_weights_from_ipc({"abc": "ipc://sock"})

        mock_update.assert_called_once()
        self.assertEqual(mock_update.call_args.args[1], "ipc://sock")
        self.assertEqual(mock_update.call_args.kwargs["device_id"], 3)
        self.assertEqual(mock_update.call_args.kwargs["run"], "loader")
        self.assertEqual(mock_update.call_args.kwargs["post_hook"], "hook")

    def test_update_weights_from_ipc_accepts_normalized_uuid(self):
        worker = DummyWorker()
        mock_update = MagicMock()

        with patch.object(worker_module, "ce_update_weights_from_ipc", mock_update):
            worker.update_weights_from_ipc({"abc": "ipc://sock"})

        self.assertEqual(mock_update.call_args.args[1], "ipc://sock")

    def test_update_weights_from_ipc_raises_when_dependency_missing(self):
        worker = DummyWorker()
        with patch.object(worker_module, "ce_update_weights_from_ipc", None):
            with self.assertRaises(ImportError):
                worker.update_weights_from_ipc({"GPU-ABC": "ipc://sock"})

    def test_impl_get_device_uuid_preserves_existing_prefix(self):
        model_runner = MagicMock()
        worker = worker_module.SGLangCheckpointEngineWorkerExtensionImpl(model_runner)

        with patch("sglang.srt.checkpoint_engine.checkpoint_engine_worker.torch.cuda.current_device", return_value=0), patch(
            "sglang.srt.checkpoint_engine.checkpoint_engine_worker.torch.cuda.get_device_properties",
            return_value=SimpleNamespace(uuid="GPU-abc"),
        ):
            self.assertEqual(worker.get_device_uuid(), "GPU-abc")

    def test_impl_get_device_uuid_prefixes_raw_uuid(self):
        model_runner = MagicMock()
        worker = worker_module.SGLangCheckpointEngineWorkerExtensionImpl(model_runner)

        with patch("sglang.srt.checkpoint_engine.checkpoint_engine_worker.torch.cuda.current_device", return_value=0), patch(
            "sglang.srt.checkpoint_engine.checkpoint_engine_worker.torch.cuda.get_device_properties",
            return_value=SimpleNamespace(uuid="abc"),
        ):
            self.assertEqual(worker.get_device_uuid(), "GPU-abc")


if __name__ == "__main__":
    unittest.main()