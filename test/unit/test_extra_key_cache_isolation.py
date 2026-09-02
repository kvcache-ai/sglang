"""Regression tests for cache_salt/extra_key radix-cache isolation."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.cache_identity import encode_cache_identity
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers import session_controller as session_controller_module
from sglang.srt.managers.io_struct import SessionParams, TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams


def _compute_openai_key(cache_salt, extra_key):
    request = SimpleNamespace(cache_salt=cache_salt, extra_key=extra_key)
    return OpenAIServingBase._compute_extra_key(None, request)


def _make_req(cache_salt=None, extra_key=None, lora_id=None):
    return Req(
        rid="test-request",
        origin_input_text="test",
        origin_input_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=1),
        extra_key=_compute_openai_key(cache_salt, extra_key),
        lora_id=lora_id,
    )


def _make_tokenized_req(extra_key, session_params=None):
    return TokenizedGenerateReqInput(
        input_text="test",
        input_ids=[1, 2, 3],
        mm_inputs=None,
        sampling_params=SamplingParams(max_new_tokens=1),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        rid="test-request",
        session_params=session_params,
        extra_key=extra_key,
        bootstrap_port=12345,
    )


def test_cache_identity_encoding_is_canonical_and_validated():
    expected = encode_cache_identity(
        "openai", cache_salt="tenant", extra_key="retrieval-context"
    )
    assert expected == encode_cache_identity(
        "openai", extra_key="retrieval-context", cache_salt="tenant"
    )

    assert _compute_openai_key(None, None) is None
    assert _compute_openai_key("", "context") == _compute_openai_key(None, "context")

    with pytest.raises(TypeError, match="cache_salt must be a string"):
        _compute_openai_key(1, None)


def test_request_cache_identities_do_not_collide():
    # The final pair is the nested-encoding collision from the previous patch.
    cases = [
        ("a", "bc", None),
        ("ab", "c", None),
        (None, "abc", None),
        ("abc", None, None),
        ("1:a", "b", None),
        (None, "a", "b"),
    ]
    reqs = [_make_req(*case) for case in cases]
    identities = [req.extra_key for req in reqs]

    assert None not in identities
    assert len(identities) == len(set(identities))
    assert _make_req().extra_key is None

    tree = RadixCache.create_simulated(page_size=1)
    tokens = [1, 2, 3, 4]
    for index, identity in enumerate(identities):
        tree.insert(
            InsertParams(
                key=RadixKey(token_ids=tokens, extra_key=identity),
                value=torch.tensor([index] * len(tokens), dtype=torch.int64),
            )
        )

    for index, identity in enumerate(identities):
        result = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(tokens + [5], extra_key=identity))
        )
        assert result.device_indices.tolist() == [index] * len(tokens)


def test_scheduler_delivers_extra_key_to_req(monkeypatch):
    captured = {}

    class StubReq:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            self.tokenizer = None

        def set_finish_with_abort(self, _message):
            pass

    monkeypatch.setattr(scheduler_module, "Req", StubReq)

    scheduler = SimpleNamespace(
        sessions={},
        server_args=SimpleNamespace(disaggregation_bootstrap_port=12345),
        model_config=SimpleNamespace(hf_eos_token_id={2}, vocab_size=100),
        disaggregation_mode=DisaggregationMode.NULL,
        enable_metrics=False,
        tokenizer=None,
        dllm_config=None,
        init_req_max_new_tokens=lambda _req: None,
        _add_request_to_queue=lambda _req: None,
    )
    recv_req = _make_tokenized_req(
        "openai-cache-identity", session_params=SessionParams(id="missing-session")
    )

    scheduler_module.Scheduler.handle_generate_request(scheduler, recv_req)

    assert captured["extra_key"] == recv_req.extra_key


def test_session_delivers_extra_key_to_req(monkeypatch):
    captured = {}

    class StubReq:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.rid = kwargs["rid"]
            self.finished_reason = None
            self.tokenizer = None

    monkeypatch.setattr(session_controller_module, "Req", StubReq)

    session = session_controller_module.Session(
        capacity_of_str_len=100, session_id="test-session"
    )
    recv_req = _make_tokenized_req(
        "openai-cache-identity", session_params=SessionParams(id=session.session_id)
    )

    session.create_req(recv_req, tokenizer=None, vocab_size=100)

    assert captured["extra_key"] == recv_req.extra_key
