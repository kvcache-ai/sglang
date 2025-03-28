from sglang.srt.disaggregation.conn import TransferEngineArgs

from concurrent.futures import ThreadPoolExecutor
import zmq

@dataclass
class MooncakeTransferEngineConfig:
    localhost_name: str
    metadata_backend: Union[str, None]
    metadata_server: str
    protocol: str
    device_name: str

    @staticmethod
    def from_file(file_path: str) -> 'MooncakeTransferEngineConfig':
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeTransferEngineConfig(
            localhost_name=config.get("localhost_name", None),
            metadata_backend=config.get("metadata_backend", None),
            metadata_server=config.get("metadata_server"),
            protocol=config.get("protocol", "rdma"),
            device_name=config.get("device_name", ""),
        )

    @staticmethod
    def load_from_env() -> 'MooncakeTransferEngineConfig':
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv('MOONCAKE_CONFIG_PATH')
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeTransferEngineConfig.from_file(config_file_path)

class MooncakeTransferEngine:
    def __init__():
        try:
            import mooncake_vllm_adaptor as mva
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector.") from e

        self.engine = mva.mooncake_vllm_adaptor()

        try:
            self.config = MooncakeTransferEngineConfig.load_from_env()
            logger.info("Mooncake Configuration loaded successfully.")
        except ValueError as e:
            logger.error(e)
            raise
        except Exception as exc:
            logger.error(
                "An error occurred while loading the configuration: %s", exc)
            raise

        self.config = MooncakeTransferEngineConfig.load_from_env()

        self.initialize(self.config.local_hostname ,
                        self.config.metadata_server, self.config.protocol,
                        self.config.device_name, self.config.metadata_backend)

    def register(self, kv_args: KVArgs):
        for kv_data_ptr, kv_data_len in zip(kv_args.kv_data_ptrs, kv_args.kv_data_lens):
            self.engine.expRegisterMemory(kv_data_ptr, kv_data_len)

     def deregister(self, kv_args: KVArgs):
        for kv_data_ptr in kv_args.kv_data_ptrs:
            self.engine.expUnregisterMemory(kv_data_ptr)

    def initialize(self, local_hostname: str, metadata_server: str,
                   protocol: str, device_name: str,
                   metadata_backend: Union[str, None]) -> None:
        """Initialize the mooncake instance."""
        if metadata_backend is None:
            self.engine.initialize(local_hostname, metadata_server, protocol,
                                   device_name)
        else:
            supported_backend = ["etcd", "redis"]
            metadata_backend = metadata_backend.lower()
            if metadata_backend not in supported_backend:
                raise ValueError(
                    "Mooncake Configuration error. `metadata_backend`"
                    f"should be one of {supported_backend}.")

            self.engine.initializeExt(local_hostname, metadata_server,
                                      protocol, device_name, metadata_backend)

    def transfer_sync(self, remote_url: str, buffer: int, peer_buffer_address: int,
                      length: int) -> int:
        """Synchronously transfer data to the specified address."""
        ret = self.engine.transferSync(remote_url, buffer,
                                       peer_buffer_address, length)
        if ret < 0:
            logger.error("Transfer Return Error")
            raise Exception("Transfer Return Error")
        return ret