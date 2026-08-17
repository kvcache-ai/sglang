"""Manual single-host Elastic EP recovery test.

Run:

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m pytest \
        test/manual/ep/test_elastic_recover.py -v -s
"""

import os
import shlex
import signal
import subprocess
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.server_fixtures.disaggregation_fixture import get_rdma_devices_args
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    try_cached_model,
)
from sglang.utils import wait_for_http_ready

TEST_MODEL = os.environ.get(
    "SGLANG_ELASTIC_RECOVER_TEST_MODEL",
    try_cached_model(DEFAULT_MODEL_NAME_FOR_TEST_MLA),
)
EP_SIZE = 8
LOCAL_EP_SIZE = 4
DIST_INIT_ADDR = os.environ.get("SGLANG_ELASTIC_RECOVER_DIST_INIT", "127.0.0.1:25555")
PRIMARY_PORT = int(os.environ.get("SGLANG_ELASTIC_RECOVER_PRIMARY_PORT", "21000"))
JOINER_PORT = int(os.environ.get("SGLANG_ELASTIC_RECOVER_JOINER_PORT", "22000"))
RECOVER_WAIT_SECONDS = float(os.environ.get("SGLANG_ELASTIC_RECOVER_WAIT_SECONDS", "5"))
RECOVER_TIMEOUT_SECONDS = float(
    os.environ.get("SGLANG_ELASTIC_RECOVER_TIMEOUT_SECONDS", "300")
)
REQUEST_TIMEOUT_SECONDS = float(
    os.environ.get("SGLANG_ELASTIC_RECOVER_REQUEST_TIMEOUT_SECONDS", "600")
)
RANDOM_SEED = int(os.environ.get("SGLANG_ELASTIC_RECOVER_RANDOM_SEED", "42"))
REDUNDANT_EXPERTS = int(
    os.environ.get("SGLANG_ELASTIC_RECOVER_REDUNDANT_EXPERTS", "72")
)
REQUIRE_HBM_RESTORE = (
    os.environ.get("SGLANG_ELASTIC_RECOVER_REQUIRE_HBM_RESTORE") == "1"
)
FAULT_RANK = os.environ.get("SGLANG_ELASTIC_RECOVER_FAULT_RANK")
ib_devices = get_rdma_devices_args()
SEMANTIC_PROMPT = "Question: Is Paris the capital of France? Answer:"
SEMANTIC_ANSWER = "yes"


def _visible_device_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [device.strip() for device in visible.split(",") if device.strip()]
    try:
        import torch

        return [str(index) for index in range(torch.cuda.device_count())]
    except Exception:
        return []


def _descendant_pids(root_pid: int) -> set[int]:
    """Return the live descendant process IDs without touching other servers."""
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat_tail = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            parent_pid = int(stat_tail[1])
        except (FileNotFoundError, IndexError, ValueError):
            continue
        children.setdefault(parent_pid, []).append(int(entry.name))

    descendants = set()
    pending = list(children.get(root_pid, []))
    while pending:
        pid = pending.pop()
        if pid in descendants:
            continue
        descendants.add(pid)
        pending.extend(children.get(pid, []))
    return descendants


def _scheduler_pid_for_rank(root_pid: int, dp_rank: int) -> int:
    """Find exactly one scheduler for a local DP rank under ``root_pid``.

    ``nvidia-smi`` reports host PIDs from this Docker configuration while this
    test sees container PIDs in ``/proc``. Scheduler process titles carry the
    displayed DP rank, so use that namespace-local identity instead.
    """
    title_prefix = f"sglang::scheduler_DP{dp_rank}_"
    descendants = _descendant_pids(root_pid)
    candidates = []
    for pid in descendants:
        try:
            title = Path(f"/proc/{pid}/cmdline").read_bytes().replace(
                b"\0", b" "
            ).decode(errors="replace")
        except FileNotFoundError:
            continue
        if title.startswith(title_prefix):
            candidates.append(pid)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one scheduler for DP rank {dp_rank} under {root_pid}; "
            f"found {candidates} among descendants {sorted(descendants)}"
        )
    return candidates[0]


def _server_args(node_rank: int, port: int, recover: bool = False) -> list[str]:
    args = [
        "sglang",
        "serve",
        "--model-path",
        TEST_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device",
        "cuda",
        "--trust-remote-code",
        "--tp",
        str(EP_SIZE),
        "--dp",
        str(EP_SIZE),
        "--nnodes",
        "2",
        "--node-rank",
        str(node_rank),
        "--dist-init-addr",
        DIST_INIT_ADDR,
        "--random-seed",
        str(RANDOM_SEED),
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--elastic-ep-backend",
        "mooncake",
        "--mooncake-ib-device",
        ib_devices,
        "--moe-a2a-backend",
        "mooncake",
        "--deepep-mode",
        "low_latency",
        "--moe-dense-tp-size",
        "1",
        "--enable-eplb",
        "--ep-num-redundant-experts",
        str(REDUNDANT_EXPERTS),
        "--chunked-prefill-size",
        "512",
        "--cuda-graph-max-bs-decode",
        "16",
        "--mem-fraction-static",
        "0.5",
        "--skip-server-warmup",
    ]
    if recover:
        args.extend(["--elastic-ep-join-mode", "recover"])
    extra_args = os.environ.get("SGLANG_ELASTIC_RECOVER_EXTRA_SERVER_ARGS", "")
    return args + shlex.split(extra_args)


@unittest.skipUnless(
    len(_visible_device_ids()) >= EP_SIZE,
    "Elastic EP recovery E2E needs 8 visible GPUs.",
)
class TestElasticRecover4To4(CustomTestCase):
    """Kill one four-rank node and recover it with a fresh process group."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = f"http://127.0.0.1:{PRIMARY_PORT}"
        cls.processes: list[subprocess.Popen] = []
        cls.log_files = []
        cls.log_paths: dict[str, Path] = {}
        visible_devices = _visible_device_ids()

        cls.primary = cls._launch(
            node_rank=0,
            port=PRIMARY_PORT,
            visible_devices=visible_devices[:LOCAL_EP_SIZE],
            name="primary",
        )
        cls.initial_joiner = cls._launch(
            node_rank=1,
            port=JOINER_PORT,
            visible_devices=visible_devices[LOCAL_EP_SIZE:EP_SIZE],
            name="initial_joiner",
        )
        wait_for_http_ready(
            f"{cls.base_url}/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.primary,
        )
        wait_for_http_ready(
            f"http://127.0.0.1:{JOINER_PORT}/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.initial_joiner,
        )
        time.sleep(RECOVER_WAIT_SECONDS)
        cls._assert_processes_alive("initial cohort readiness")

    @classmethod
    def _launch(
        cls,
        *,
        node_rank: int,
        port: int,
        visible_devices: list[str],
        name: str,
        recover: bool = False,
    ) -> subprocess.Popen:
        log_path = Path(f"/tmp/elastic_ep_recover_{name}_{int(time.time())}.log")
        log_file = open(log_path, "w")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
        # The generic health endpoint normally submits a one-token generation
        # while polling. The test establishes its inference baseline explicitly
        # after both nodes are ready.
        env["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] = "0"
        process = subprocess.Popen(
            _server_args(node_rank, port, recover),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        cls.processes.append(process)
        cls.log_files.append(log_file)
        cls.log_paths[name] = log_path
        print(f"Started {name}; log: {log_path}")
        return process

    @classmethod
    def _assert_processes_alive(cls, description: str) -> None:
        for name, process in zip(cls.log_paths, cls.processes):
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"{description}: {name} server process exited with {exit_code}"
                )

    @classmethod
    def tearDownClass(cls):
        for process in reversed(getattr(cls, "processes", [])):
            if process.poll() is None:
                kill_process_tree(process.pid, wait_timeout=60)
        for log_file in getattr(cls, "log_files", []):
            log_file.close()

    def _generate(
        self,
        routed_dp_rank: int | None = None,
        *,
        fault_probe: bool = False,
        base_url: str | None = None,
    ) -> requests.Response:
        payload = {
            "text": SEMANTIC_PROMPT,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0.0,
                "regex": "( Yes| No)",
            },
        }
        if fault_probe:
            # Diverse token IDs force a prefill through a broad expert set, so
            # every surviving DP scheduler contacts the killed EP peer.
            payload["input_ids"] = list(range(256, 512))
            payload.pop("text")
        if routed_dp_rank is not None:
            payload["routed_dp_rank"] = routed_dp_rank
        return requests.post(
            f"{base_url or self.base_url}/generate",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    def _generate_ok(
        self,
        description: str,
        routed_dp_rank: int | None = None,
        *,
        base_url: str | None = None,
    ) -> None:
        response = self._generate(routed_dp_rank, base_url=base_url)
        self.assertEqual(response.status_code, 200, f"{description}: {response.text}")
        payload = response.json()
        generated_text = payload.get("text", "")
        self.assertTrue(
            generated_text.strip(),
            f"{description}: empty generation response: {payload!r}",
        )
        self.assertIn(
            SEMANTIC_ANSWER,
            generated_text.lower(),
            f"{description}: semantic answer is incorrect: {payload!r}",
        )

    def _generate_all_ok(
        self,
        description: str,
        targets: list[tuple[str, int | None]],
        *,
        fault_probe: bool = False,
        allow_request_errors: bool = False,
        require_semantics: bool = True,
    ) -> None:
        """Drive every local scheduler concurrently through one generation."""
        with ThreadPoolExecutor(max_workers=len(targets)) as executor:
            futures = {
                (base_url, routed_dp_rank): executor.submit(
                    self._generate,
                    routed_dp_rank=routed_dp_rank,
                    fault_probe=fault_probe,
                    base_url=base_url,
                )
                for base_url, routed_dp_rank in targets
            }
            for (base_url, routed_dp_rank), future in futures.items():
                try:
                    response = future.result()
                except requests.RequestException as exc:
                    if allow_request_errors:
                        print(
                            f"{description} on {base_url} DP rank "
                            f"{routed_dp_rank}: {exc}"
                        )
                        continue
                    self.fail(
                        f"{description} on {base_url} DP rank "
                        f"{routed_dp_rank}: {exc}"
                    )
                if response.status_code != 200:
                    if allow_request_errors:
                        print(
                            f"{description} on {base_url} DP rank "
                            f"{routed_dp_rank}: HTTP {response.status_code}: "
                            f"{response.text}"
                        )
                        continue
                    self.fail(
                        f"{description} on {base_url} DP rank {routed_dp_rank}: "
                        f"{response.text}"
                    )
                if require_semantics:
                    self.assertTrue(
                        response.json().get("text", "").strip(),
                        f"{description} on {base_url} DP rank {routed_dp_rank}: "
                        f"empty generation response: {response.text}",
                    )
                    self.assertIn(
                        SEMANTIC_ANSWER,
                        response.json()["text"].lower(),
                        f"{description} on {base_url} DP rank {routed_dp_rank}: "
                        f"semantic answer is incorrect: {response.text}",
                    )

    def _wait_for_recover_capture(self) -> None:
        if os.environ.get("SGLANG_ELASTIC_RECOVER_SKIP_CUDA_GRAPH_WAIT") == "1":
            return
        deadline = time.monotonic() + RECOVER_TIMEOUT_SECONDS
        log_path = self.log_paths["recover_joiner"]
        marker = "Capture target decode CUDA graph end"
        while time.monotonic() < deadline:
            self.assertIsNone(
                self.recover_joiner.poll(),
                "Recover joiner exited during CUDA graph capture",
            )
            if log_path.exists() and log_path.read_text(errors="replace").count(
                marker
            ) >= (EP_SIZE - LOCAL_EP_SIZE):
                return
            time.sleep(2)
        self.fail(f"Timed out waiting for recover CUDA graph capture: {log_path}")

    def _wait_for_recovered_ranks(self) -> None:
        self._wait_for_recover_capture()
        self._generate_ok("recovery trigger")
        deadline = time.monotonic() + RECOVER_TIMEOUT_SECONDS
        marker = f"recover ranks {list(range(LOCAL_EP_SIZE, EP_SIZE))} done"
        primary_log = self.log_paths["primary"]
        while time.monotonic() < deadline:
            self.assertIsNone(
                self.recover_joiner.poll(), "Recover joiner exited before rejoining"
            )
            if (
                primary_log.exists()
                and primary_log.read_text(errors="replace").count(marker)
                >= LOCAL_EP_SIZE
            ):
                for request_index in range(3):
                    self._generate_ok(f"post-recovery request {request_index + 1}")
                return
            time.sleep(2)
        self.fail(f"Timed out waiting for recovery collective: {primary_log}")

    def _wait_for_hbm_restore(self) -> None:
        if not REQUIRE_HBM_RESTORE:
            return
        deadline = time.monotonic() + RECOVER_TIMEOUT_SECONDS
        primary_log = self.log_paths["primary"]
        marker = "[DaemonHBMExpertRecovery]"
        while time.monotonic() < deadline:
            self.assertIsNone(
                self.primary.poll(),
                "Primary exited while waiting for daemon-HBM restore",
            )
            if primary_log.exists() and marker in primary_log.read_text(errors="replace"):
                return
            time.sleep(2)
        self.fail(f"Timed out waiting for daemon-HBM restore: {primary_log}")

    def _kill_fault_rank(self) -> None:
        assert FAULT_RANK is not None
        fault_rank = int(FAULT_RANK)
        self.assertIn(fault_rank, range(LOCAL_EP_SIZE))
        scheduler_pid = _scheduler_pid_for_rank(self.primary.pid, fault_rank)
        print(f"Killing primary scheduler DP rank {fault_rank}, pid {scheduler_pid}")
        os.kill(scheduler_pid, signal.SIGKILL)
        print(f"Killed primary scheduler for global rank {fault_rank}")

    def test_recover_four_ranks(self):
        primary_targets = [(self.base_url, rank) for rank in range(LOCAL_EP_SIZE)]
        # Mooncake EP establishes peer transports on the first data-plane
        # forwards. This deliberately non-semantic round lets the initial
        # cohort settle before the baseline serving assertion.
        self._generate_all_ok(
            "initial EP transport prime",
            primary_targets,
            fault_probe=True,
            allow_request_errors=True,
            require_semantics=False,
        )
        self._generate_all_ok("initial service", primary_targets)

        if FAULT_RANK is None:
            kill_process_tree(self.initial_joiner.pid, wait_timeout=60)
        else:
            self._kill_fault_rank()
        # Give the terminated schedulers time to disappear before fault handling.
        time.sleep(RECOVER_WAIT_SECONDS)
        # Rebalancing is demand-driven. A rank fault must be observed by every
        # remaining local scheduler before their independent EPLB managers can
        # install the same new placement.
        if FAULT_RANK is not None:
            # The initial second node is part of the serving EP cohort but is
            # not an externally routed controller. Drive every surviving
            # primary-controller rank; EPLB remains an EP-global collective.
            trigger_targets = [
                (self.base_url, rank)
                for rank in range(LOCAL_EP_SIZE)
                if rank != int(FAULT_RANK)
            ]
        else:
            trigger_targets = [(self.base_url, None)]
        # EPLB recovery is collective across the surviving schedulers. Send
        # every rank's first post-fault request concurrently so the first
        # request cannot block the other ranks from entering the collective.
        self._generate_all_ok(
            "fault recovery trigger",
            trigger_targets,
            fault_probe=FAULT_RANK is not None,
            allow_request_errors=FAULT_RANK is not None,
            require_semantics=FAULT_RANK is None,
        )
        self._wait_for_hbm_restore()
        degraded_targets = [
            (self.base_url, rank)
            for rank in range(LOCAL_EP_SIZE)
            if FAULT_RANK is None or rank != int(FAULT_RANK)
        ]
        self._generate_all_ok("degraded service after node1 failure", degraded_targets)

        if FAULT_RANK is not None:
            return

        visible_devices = _visible_device_ids()
        self.recover_joiner = self._launch(
            node_rank=1,
            port=JOINER_PORT,
            visible_devices=visible_devices[LOCAL_EP_SIZE:EP_SIZE],
            name="recover_joiner",
            recover=True,
        )
        self._wait_for_recovered_ranks()


if __name__ == "__main__":
    unittest.main()
