import os
import subprocess
import sys
import textwrap
import unittest


class TestGlm4MoeRopeConfig(unittest.TestCase):
    def test_transformers_v5_rope_parameters_reach_attention(self):
        script = textwrap.dedent("""
            import importlib.util
            import sys
            import types
            from types import SimpleNamespace
            from unittest.mock import MagicMock, patch

            import torch

            torch.cuda.get_device_capability = lambda *args, **kwargs: (8, 9)
            spec = importlib.util.find_spec("sgl_kernel")
            fake_kernel = types.ModuleType("sgl_kernel")
            fake_kernel.__path__ = list(spec.submodule_search_locations)
            fake_kernel.__file__ = spec.origin
            fake_kernel.__spec__ = spec
            fake_kernel.common_ops = SimpleNamespace()

            def missing_kernel(name):
                if name.startswith("__"):
                    raise AttributeError(name)
                return lambda *args, **kwargs: None

            fake_kernel.__getattr__ = missing_kernel
            sys.modules["sgl_kernel"] = fake_kernel

            from sglang.srt.models.glm4_moe import Glm4MoeDecoderLayer

            class CaptureAttention(torch.nn.Module):
                kwargs = None

                def __init__(self, **kwargs):
                    super().__init__()
                    type(self).kwargs = kwargs

            class StubModule(torch.nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()

            config = SimpleNamespace(
                hidden_size=8,
                num_attention_heads=2,
                num_key_value_heads=2,
                num_hidden_layers=1,
                intermediate_size=16,
                hidden_act="silu",
                rms_norm_eps=1e-5,
                attention_bias=False,
                n_routed_experts=None,
                rope_parameters={
                    "rope_theta": 1_000_000,
                    "partial_rotary_factor": 0.25,
                    "rope_type": "default",
                },
            )

            with (
                patch("sglang.srt.models.glm4_moe.Glm4MoeAttention", CaptureAttention),
                patch("sglang.srt.models.glm4_moe.Glm4MoeMLP", StubModule),
                patch("sglang.srt.models.glm4_moe.RMSNorm", StubModule),
                patch(
                    "sglang.srt.models.glm4_moe.enable_moe_dense_fully_dp",
                    return_value=False,
                ),
                patch(
                    "sglang.srt.models.glm4_moe.LayerScatterModes.init_new",
                    return_value=MagicMock(),
                ),
                patch("sglang.srt.models.glm4_moe.LayerCommunicator", MagicMock),
            ):
                Glm4MoeDecoderLayer(config, layer_id=0)

            assert CaptureAttention.kwargs["rope_theta"] == 1_000_000
            assert CaptureAttention.kwargs["rope_scaling"] == config.rope_parameters
            assert CaptureAttention.kwargs["partial_rotary_factor"] == 0.25
            """)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            ),
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
