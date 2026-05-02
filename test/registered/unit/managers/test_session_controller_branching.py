import unittest
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import FINISH_LENGTH
from sglang.srt.session.session_controller import DEFAULT_SESSION_BRANCH_ID, Session
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")


class TestSessionControllerBranching(CustomTestCase):
    def _make_req(self, rid: str, input_ids, *, branch_id=None, fork_from_rid=None):
        return SimpleNamespace(
            rid=rid,
            input_ids=list(input_ids),
            mm_inputs=None,
            sampling_params=SimpleNamespace(max_new_tokens=2, custom_params={}),
            lora_id=None,
            custom_logit_processor=None,
            stream=False,
            return_logprob=False,
            top_logprobs_num=0,
            token_ids_logprob=[],
            require_reasoning=False,
            return_hidden_states=False,
            return_routed_experts=False,
            priority=None,
            routing_key=None,
            extra_key=None,
            http_worker_ipc=None,
            time_stats=None,
            session_params=SimpleNamespace(
                id="session-a",
                rid=None,
                offset=None,
                replace=False,
                drop_previous_output=False,
                agent_id="agent-1",
                branch_id=branch_id,
                fork_from_rid=fork_from_rid,
                cache_scope="branch",
            ),
        )

    def test_streaming_branching_tracks_branch_heads(self):
        session = Session(
            1024,
            session_id="session-a",
            streaming=True,
            agent_id="agent-1",
            enable_agentic_branching=True,
        )

        req1 = session.create_req(self._make_req("r1", [1, 2, 3]), None, 128)
        req1.output_ids = [10, 11]
        req1.finished_reason = FINISH_LENGTH(2)
        session.finish_req(req1)

        req2 = session.create_req(
            self._make_req("r2", [4, 5], branch_id="plan", fork_from_rid="r1"),
            None,
            128,
        )
        req2.output_ids = [12]
        req2.finished_reason = FINISH_LENGTH(1)
        session.finish_req(req2)

        req3 = session.create_req(
            self._make_req("r3", [6], branch_id="plan"),
            None,
            128,
        )
        req3.finished_reason = FINISH_LENGTH(0)
        session.finish_req(req3)

        self.assertEqual(req1.session_branch_id, DEFAULT_SESSION_BRANCH_ID)
        self.assertEqual(req2.session_branch_id, "plan")
        self.assertEqual(req2.session_fork_from_rid, "r1")
        self.assertEqual(req2.session_agent_id, "agent-1")
        self.assertEqual(req2.session_cache_scope, "branch")
        self.assertEqual(req2.origin_input_ids, [1, 2, 3, 10, 11, 4, 5])
        self.assertEqual(req3.origin_input_ids, [1, 2, 3, 10, 11, 4, 5, 12, 6])
        self.assertEqual(session.branches["plan"].head_rid, "r3")
        self.assertEqual(session.branches["plan"].forked_from_rid, "r1")


if __name__ == "__main__":
    unittest.main()