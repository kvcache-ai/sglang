"""End-to-end tests for branch-aware streaming sessions."""

import os
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=75, suite="stage-b-test-1-gpu-large")


def _generate(base_url, input_ids, *, session_params=None, max_new_tokens=12):
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "no_stop_trim": True,
            "skip_special_tokens": False,
        },
    }
    if session_params is not None:
        payload["session_params"] = session_params

    response = requests.post(base_url + "/generate", json=payload, timeout=120)
    assert response.status_code == 200, response.text
    return response.json()


class TestStreamingSessionBranching(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get(
            "SGLANG_TEST_MODEL", DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-streaming-session",
                "--chunked-prefill-size",
                "512",
                "--attention-backend",
                "triton",
            ],
        )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_streaming_session_branch_slots_are_reused_per_branch(self):
        requests.post(self.base_url + "/flush_cache", timeout=30)
        open_resp = requests.post(
            self.base_url + "/open_session",
            json={
                "capacity_of_str_len": 4096,
                "streaming": True,
                "agent_id": "agent-1",
                "enable_agentic_branching": True,
                "cache_scope": "branch",
            },
            timeout=30,
        )
        self.assertEqual(open_resp.status_code, 200)
        session_id = open_resp.json()

        root_ids = self.tokenizer.encode("Let me tell you something about France.")
        plan_ids = self.tokenizer.encode("The capital of France is")
        plan_followup_ids = self.tokenizer.encode("Its most famous museum is")
        tool_ids = self.tokenizer.encode("The population of the city is")
        tool_followup_ids = self.tokenizer.encode("Its metropolitan area is")

        for input_ids in (plan_ids, plan_followup_ids, tool_ids, tool_followup_ids):
            if input_ids and input_ids[0] == self.tokenizer.bos_token_id:
                del input_ids[0]

        root_resp = _generate(
            self.base_url,
            root_ids,
            session_params={"id": session_id},
        )
        root_rid = root_resp["meta_info"]["id"]
        self.assertEqual(root_resp["meta_info"]["cached_tokens"], 0)

        plan_first = _generate(
            self.base_url,
            plan_ids,
            session_params={
                "id": session_id,
                "branch_id": "plan",
                "fork_from_rid": root_rid,
                "agent_id": "agent-1",
                "cache_scope": "branch",
            },
        )
        plan_first_kv = (
            plan_first["meta_info"]["prompt_tokens"]
            + plan_first["meta_info"]["completion_tokens"]
        )

        plan_second = _generate(
            self.base_url,
            plan_followup_ids,
            session_params={
                "id": session_id,
                "branch_id": "plan",
                "agent_id": "agent-1",
                "cache_scope": "branch",
            },
        )
        self.assertEqual(plan_second["meta_info"]["cached_tokens"], plan_first_kv)

        tool_first = _generate(
            self.base_url,
            tool_ids,
            session_params={
                "id": session_id,
                "branch_id": "tool",
                "fork_from_rid": root_rid,
                "agent_id": "agent-1",
                "cache_scope": "branch",
            },
        )
        tool_first_kv = (
            tool_first["meta_info"]["prompt_tokens"]
            + tool_first["meta_info"]["completion_tokens"]
        )

        tool_second = _generate(
            self.base_url,
            tool_followup_ids,
            session_params={
                "id": session_id,
                "branch_id": "tool",
                "agent_id": "agent-1",
                "cache_scope": "branch",
            },
        )
        self.assertEqual(tool_second["meta_info"]["cached_tokens"], tool_first_kv)

        plan_resume_ids = self.tokenizer.encode("The river through the city is")
        if plan_resume_ids and plan_resume_ids[0] == self.tokenizer.bos_token_id:
            del plan_resume_ids[0]

        plan_resume = _generate(
            self.base_url,
            plan_resume_ids,
            session_params={
                "id": session_id,
                "branch_id": "plan",
                "agent_id": "agent-1",
                "cache_scope": "branch",
            },
        )
        plan_second_kv = (
            plan_second["meta_info"]["prompt_tokens"]
            + plan_second["meta_info"]["completion_tokens"]
        )
        self.assertEqual(plan_resume["meta_info"]["cached_tokens"], plan_second_kv)

        close_resp = requests.post(
            self.base_url + "/close_session",
            json={"session_id": session_id},
            timeout=30,
        )
        self.assertEqual(close_resp.status_code, 200)

        time.sleep(2)
        health_resp = requests.get(self.base_url + "/health", timeout=30)
        self.assertEqual(health_resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()