"""CPU tokenizer tests; these do not replace server generation acceptance."""

import ast
import os
from pathlib import Path
from typing import Optional
import unittest
from unittest.mock import Mock

from llguidance import LLMatcher, LLTokenizer
from llguidance.hf import from_tokenizer
import tiktoken

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/constrained/llguidance_backend.py"
)


def load_factory(hf_factory=from_tokenizer):
    # Isolate the production factory from unrelated CUDA/server imports.
    tree = ast.parse(SOURCE.read_bytes())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_create_llguidance_tokenizer"
    )
    namespace = {
        "Optional": Optional,
        "LLTokenizer": LLTokenizer,
        "from_tokenizer": hf_factory,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace[function.name]


def kimi_fixture():
    tokenizer_type = type("TikTokenTokenizer", (), {"__module__": "tokenization_kimi"})
    tokenizer = tokenizer_type()
    tokenizer.is_fast = False
    tokenizer.eos_token_id = 257
    tokenizer.model = tiktoken.Encoding(
        name="kimi-unit",
        pat_str=r"(?s).",
        mergeable_ranks={bytes([value]): value for value in range(256)},
        special_tokens={"<|endoftext|>": 256, "[EOS]": 257},
    )
    return tokenizer


def accepts(tokenizer, text):
    schema = '{"type":"object","properties":{"answer":{"const":"喵"}},"required":["answer"],"additionalProperties":false}'
    matcher = LLMatcher(tokenizer, LLMatcher.grammar_from_json_schema(schema))
    for token in tokenizer.tokenize_str(text):
        if not matcher.consume_token(token):
            return False
    return (
        matcher.consume_token(tokenizer.eos_token)
        and matcher.is_stopped()
        and not matcher.is_error()
    )


def assert_special_tokens(test, tokenizer, native):
    # llguidance parses <...> literals only; generation consumes token IDs.
    for text, token_id in tokenizer.model._special_tokens.items():
        with test.subTest(special_token=text, token_id=token_id):
            test.assertTrue(native.is_special_token(token_id))
            test.assertEqual(native.decode_bytes([token_id]), text.encode("utf-8"))


class KimiGuidanceTokenizerTest(unittest.TestCase):
    def test_tokenization_eos_and_padded_vocabulary(self):
        tokenizer = kimi_fixture()
        native = load_factory()(tokenizer, 260)
        self.assertEqual(native.eos_token, 257)
        self.assertEqual(native.vocab_size, 260)
        self.assertFalse(tokenizer.is_fast)
        for text in ("你好，世界！", "Hello world", "12345", "a\nb"):
            with self.subTest(text=text):
                expected = tokenizer.model.encode_ordinary(text)
                self.assertEqual(native.tokenize_str(text), expected)
        assert_special_tokens(self, tokenizer, native)

    def test_json_schema_accepts_and_rejects(self):
        native = load_factory()(kimi_fixture(), None)
        self.assertTrue(accepts(native, '{"answer":"喵"}'))
        self.assertFalse(accepts(native, '{"answer":"汪"}'))

    def test_other_tokenizers_keep_hf_dispatch(self):
        hf_factory = Mock(return_value=object())
        tokenizer = object()
        self.assertIs(load_factory(hf_factory)(tokenizer, 300), hf_factory.return_value)
        hf_factory.assert_called_once_with(tokenizer, 300)

    def test_foreign_slow_tokenizer_is_not_misclassified(self):
        tokenizer = kimi_fixture()
        tokenizer.__class__ = type(
            "TikTokenTokenizer", (), {"__module__": "another_model"}
        )
        with self.assertRaisesRegex(ValueError, "Only fast tokenizers"):
            load_factory()(tokenizer, None)

    def test_invalid_encoding_does_not_use_kimi_dispatch(self):
        tokenizer = kimi_fixture()
        tokenizer.model = {}
        with self.assertRaisesRegex(ValueError, "Only fast tokenizers"):
            load_factory()(tokenizer, None)

    def test_hf_fast_tokenizer(self):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        vocab = {
            char: index
            for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
        }
        vocab["<|endoftext|>"] = len(vocab)
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, eos_token="<|endoftext|>"
        )
        native = load_factory()(tokenizer, len(vocab))
        self.assertEqual(
            native.tokenize_str("hello world"),
            tokenizer.encode("hello world", add_special_tokens=False),
        )

    @unittest.skipUnless(
        os.environ.get("KT_KIMI_TOKENIZER_PATH"), "Set path to official Kimi tokenizer"
    )
    def test_official_kimi_tokenizer(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            os.environ["KT_KIMI_TOKENIZER_PATH"],
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
        native = load_factory()(tokenizer, len(tokenizer))
        self.assertEqual(native.eos_token, tokenizer.eos_token_id)
        self.assertEqual(native.vocab_size, len(tokenizer))
        for text in (
            "你好，世界！",
            "Hello, world!",
            "1234567890",
            "第一行\n第二行\n",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    native.tokenize_str(text),
                    tokenizer.encode(text, add_special_tokens=False),
                )
        assert_special_tokens(self, tokenizer, native)
        self.assertTrue(accepts(native, '{"answer":"喵"}'))
        self.assertFalse(accepts(native, '{"answer":"汪"}'))


if __name__ == "__main__":
    unittest.main()
