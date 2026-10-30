import importlib.util
import os
import shlex
import subprocess
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.server_fixtures.disaggregation_fixture import get_rdma_devices_args
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)
from sglang.utils import wait_for_http_ready

register_cuda_ci(est_time=526, stage="extra-b", runner_config="8-gpu-h200")

DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2"

ib_devices = get_rdma_devices_args()


def _load_sibling_test_module(module_name: str, filename: str):
    module_path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load test module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TestDeepseekMTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--disable-overlap-schedule",
                "--trust-remote-code",
                "--tp",
                "8",
                "--attn-dp-size",
                "8",
                "--moe-dense-tp-size",
                "1",
                "--enable-dp-lm-head",
                "--moe-a2a-backend",
                "deepep",
                "--moe-runner-backend",
                "deep_gemm",
                "--enable-two-batch-overlap",
                "--ep-num-redundant-experts",
                "32",
                "--ep-dispatch-algorithm",
                "dynamic",
                "--eplb-algorithm",
                "deepseek",
                "--cuda-graph-bs-decode",
                "64",  # TODO: increase it to 128 when TBO is supported in draft_extend
                "--max-running-requests",
                "512",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "2",
                "--disable-radix-cache",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true,"num_threads": 64}',
            ],
        )

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            max_tokens=512,
            num_examples=1200,
            num_threads=1200,
        )
        metrics = run_sgl_eval(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")

        self.assertGreater(metrics["score"], 0.92)

        server_info = requests.get(self.base_url + "/server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(
            f"###test_gsm8k:\n"
            f"accuracy={metrics['score']=:.3f}\n"
            f"{avg_spec_accept_length=:.3f}\n"
        )
        self.assertGreater(avg_spec_accept_length, 1.85)


class TestDeepseekV32TBO(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V32_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--attn-dp-size",
            "8",
            "--enable-two-batch-overlap",
            "--moe-a2a-backend",
            "deepep",
            "--cuda-graph-max-bs-decode",
            "256",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process)

    def test_a_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            max_tokens=512,
            num_examples=1200,
            num_threads=1200,
        )
        metrics = run_sgl_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.92)

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{speed=:.2f}")


# Keep the recovery diagnostic in this registered module for fast CI reruns.
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
RANDOM_SEED = int(os.environ.get("SGLANG_ELASTIC_RECOVER_RANDOM_SEED", "42"))


def _visible_device_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [device.strip() for device in visible.split(",") if device.strip()]
    try:
        import torch

        return [str(index) for index in range(torch.cuda.device_count())]
    except Exception:
        return []


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
        "--attn-dp-size",
        str(EP_SIZE),
        "--nnodes",
        "2",
        "--node-rank",
        str(node_rank),
        "--dist-init-addr",
        DIST_INIT_ADDR,
        "--random-seed",
        str(RANDOM_SEED),
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
        "--disable-custom-all-reduce",
        "--enable-eplb",
        "--ep-num-redundant-experts",
        "72",
        "--chunked-prefill-size",
        "512",
        "--cuda-graph-max-bs-decode",
        "16",
        "--mem-fraction-static",
        "0.5",
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
        try:
            wait_for_http_ready(
                f"{cls.base_url}/health_generate",
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                process=cls.primary,
            )
        except Exception:
            cls._print_server_logs()
            raise

    @classmethod
    def _print_server_logs(cls) -> None:
        """Expose child-server failures in CI, where /tmp is otherwise hidden."""
        for name, log_path in cls.log_paths.items():
            print(f"\n===== {name} log: {log_path} =====", flush=True)
            if log_path.exists():
                content = log_path.read_text(errors="replace")
                print(content[-20000:], flush=True)
            process = getattr(cls, name, None)
            if process is not None:
                print(f"{name} returncode: {process.poll()}", flush=True)

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
        log_file = open(log_path, "w", buffering=1)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
        env["PYTHONUNBUFFERED"] = "1"
        if is_in_ci():
            env["MC_DEVICE_API_P2P_ALLOW_CROSS_LOGICAL_NODE"] = "1"
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
    def tearDownClass(cls):
        for process in reversed(getattr(cls, "processes", [])):
            if process.poll() is None:
                kill_process_tree(process.pid, wait_timeout=60)
        for log_file in getattr(cls, "log_files", []):
            log_file.close()

    def _generate(self, routed_dp_rank: int | None = None) -> requests.Response:
        payload = {
            "text": "The capital of France is",
            "sampling_params": {"max_new_tokens": 4, "temperature": 0.0},
        }
        if routed_dp_rank is not None:
            payload["routed_dp_rank"] = routed_dp_rank
        return requests.post(f"{self.base_url}/generate", json=payload, timeout=90)

    def _generate_ok(self, description: str, routed_dp_rank: int | None = None) -> None:
        response = self._generate(routed_dp_rank)
        self.assertEqual(response.status_code, 200, f"{description}: {response.text}")
        payload = response.json()
        generated_text = payload.get("text", "")
        self.assertIn(
            "paris",
            generated_text.casefold(),
            f"{description}: unexpected generation: {generated_text!r}",
        )

    def _wait_for_recover_capture(self) -> None:
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

    def test_recover_four_ranks(self):
        self._generate_ok("initial service")

        kill_process_tree(self.initial_joiner.pid, wait_timeout=60)
        # Give the terminated schedulers time to disappear before fault handling.
        time.sleep(RECOVER_WAIT_SECONDS)
        self._generate_ok("degraded service after node1 failure")

        visible_devices = _visible_device_ids()
        self.recover_joiner = self._launch(
            node_rank=1,
            port=JOINER_PORT,
            visible_devices=visible_devices[LOCAL_EP_SIZE:EP_SIZE],
            name="recover_joiner",
            recover=True,
        )
        try:
            self._wait_for_recovered_ranks()
        except Exception:
            self._print_server_logs()
            raise


# Reuse this registered 8-GPU entry point for fast /rerun-test diagnostics.
_elastic_scale_module = _load_sibling_test_module(
    "sglang_registered_elastic_scale", "test_elastic_scale.py"
)
_nixl_ep_module = _load_sibling_test_module(
    "sglang_registered_nixl_ep", "test_nixl_ep.py"
)


@unittest.skipIf(is_in_ci(), "NIXL EP CI artifact is currently ABI-incompatible.")
class TestElasticScaleUp4To5To6CudaGraph(
    _elastic_scale_module.TestElasticScaleUp4To5To6CudaGraph
):
    pass


@unittest.skipIf(is_in_ci(), "NIXL EP CI artifact is currently ABI-incompatible.")
class TestNixlMoeMooncakeElasticEP(_nixl_ep_module.TestNixlMoeMooncakeElasticEP):
    pass


if __name__ == "__main__":
    unittest.main()
