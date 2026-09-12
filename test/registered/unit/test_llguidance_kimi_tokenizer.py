"""Kimi native tokenization and unchanged Hugging Face dispatch."""

from unittest.mock import Mock

import pytest
import tiktoken

from sglang.srt.constrained import llguidance_backend as guidance


def kimi_tokenizer():
    cls = type("TikTokenTokenizer", (), {"__module__": "tokenization_kimi"})
    tokenizer = cls()
    tokenizer.eos_token_id = 256
    tokenizer.model = tiktoken.Encoding(
        name="kimi-test",
        pat_str=r"(?s).",
        mergeable_ranks={bytes([value]): value for value in range(256)},
        special_tokens={"[EOS]": 256},
    )
    return tokenizer


def test_kimi_preserves_tokens_eos_and_vocabulary():
    tokenizer = kimi_tokenizer()
    native = guidance._create_llguidance_tokenizer(tokenizer, 260)
    assert native.eos_token == 256
    assert native.vocab_size == 260
    assert native.decode_bytes([256]) == b"[EOS]"
    for text in ("Hello", "你好，世界！", "123\n456"):
        assert native.tokenize_str(text) == tokenizer.model.encode_ordinary(text)


@pytest.mark.parametrize("kind", ["ordinary", "foreign", "invalid_encoding"])
def test_other_tokenizers_keep_existing_dispatch(monkeypatch, kind):
    tokenizer = kimi_tokenizer()
    if kind == "ordinary":
        tokenizer = object()
    elif kind == "foreign":
        tokenizer.__class__ = type(
            "TikTokenTokenizer", (), {"__module__": "other_model"}
        )
    else:
        tokenizer.model = None
    fallback = Mock(return_value=object())
    monkeypatch.setattr(guidance, "from_tokenizer", fallback)
    assert (
        guidance._create_llguidance_tokenizer(tokenizer, 260) is fallback.return_value
    )
    fallback.assert_called_once_with(tokenizer, 260)
