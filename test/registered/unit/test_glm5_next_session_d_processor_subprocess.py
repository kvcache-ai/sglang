"""Import smoke for the split transformers-kt/SGLang GLM-5-Next processor."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSOR_PATH = REPO_ROOT / "python/sglang/srt/multimodal/processors/glm5_next.py"


def test_production_processor_imports_transformers_kt_glm5_classes():
    child = textwrap.dedent(
        f"""
        import importlib.util
        import sys
        import types

        from transformers.models.glm5_next.image_processing_glm5_next import Glm5NextImageProcessor
        from transformers.models.glm5_next.processing_glm5_next import Glm5NextProcessor
        from transformers.models.glm5_next.video_processing_glm5_next import Glm5NextVideoProcessor

        def package(name):
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
            return module

        def module(name, **attrs):
            value = types.ModuleType(name)
            value.__dict__.update(attrs)
            sys.modules[name] = value
            return value

        for name in (
            "sglang",
            "sglang.srt",
            "sglang.srt.layers",
            "sglang.srt.models",
            "sglang.srt.multimodal",
            "sglang.srt.multimodal.processors",
        ):
            package(name)

        class Stub:
            pass

        class BaseOutput:
            def __init__(self, input_text, images=None, videos=None, audios=None):
                self.input_text = input_text
                self.images = images or []
                self.videos = videos or []
                self.audios = audios or []

        module("sglang.srt.layers.rotary_embedding", MRotaryEmbedding=Stub)
        module(
            "sglang.srt.models.glm5_next",
            Glm5NextForConditionalGeneration=Stub,
        )
        module(
            "sglang.srt.multimodal.processors.base_processor",
            BaseMultiModalProcessorOutput=BaseOutput,
            MultimodalSpecialTokens=Stub,
        )
        module(
            "sglang.srt.multimodal.processors.glm4v",
            Glm4vImageProcessor=Stub,
        )

        spec = importlib.util.spec_from_file_location(
            "sglang.srt.multimodal.processors.glm5_next",
            {str(PROCESSOR_PATH)!r},
        )
        production = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = production
        spec.loader.exec_module(production)

        assert production.Glm5NextImageProcessor is Glm5NextImageProcessor
        assert production.Glm5NextProcessor is Glm5NextProcessor
        assert production.Glm5NextVideoProcessor is Glm5NextVideoProcessor
        assert production.GLM5_NEXT_PATCH_EXPAND_FACTOR == 1
        assert production.GLM5_NEXT_DEFAULT_MAX_PIXELS == 6_272_000
        print("GLM5_VIDEO_PROCESSOR_IMPORT_OK")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", child],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise AssertionError(
            f"child import failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    assert "GLM5_VIDEO_PROCESSOR_IMPORT_OK" in completed.stdout
