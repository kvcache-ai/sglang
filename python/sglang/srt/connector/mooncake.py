# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Generator, List, Optional, Tuple

import torch

from sglang.srt.connector import BaseKVConnector
from sglang.srt.connector.serde import create_serde
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    MooncakeStoreConfig,
)

logger = logging.getLogger(__name__)


class MooncakeConnector(BaseKVConnector):
    def __init__(self, url: str):
        super().__init__(url)
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://kvcache-ai.github.io/Mooncake/getting_started/build.html"
                "to run SGLang with MooncakeConnector."
            ) from e

        # Load configuration
        # We try to load from environment variables first as it is the most common way
        # for connectors.
        # TODO: Support loading from config file specified in url or other means if needed.
        try:
            self.config = MooncakeStoreConfig.load_from_env()
        except Exception:
            # Fallback to defaults or try to load from file if env vars are missing
            # But MooncakeStoreConfig.load_from_env() raises ValueError if essential envs are missing.
            # We might want to allow partial config or defaults.
            # For now, let's assume the user has set up the environment variables as per Mooncake docs.
            logger.warning(
                "Failed to load Mooncake config from environment variables. "
                "Attempting to load from default config path."
            )
            try:
                self.config = MooncakeStoreConfig.from_file()
            except Exception as e:
                raise RuntimeError(
                    "Failed to initialize MooncakeConnector. "
                    "Please set MOONCAKE_MASTER/MOONCAKE_CLIENT environment variables "
                    "or provide a valid config file."
                ) from e

        self.store = MooncakeDistributedStore()
        
        # Setup the store
        # We need to determine if we are a client or a full node.
        # Reusing logic from MooncakeStore
        
        # We use a dummy buffer size for setup as we might not be using the memory pool here
        # strictly for serving, but just for connector access.
        # However, Mooncake setup requires some buffer size.
        # Let's use a small buffer or 0 if allowed.
        local_buffer_size = 0 
        
        # Check if we are in standalone/client mode
        if self.config.standalone_storage:
             ret_code = self.store.setup_dummy(
                0, # global_segment_size (0 for client only?)
                local_buffer_size,
                self.config.client_server_address,
            )
        else:
            # Full node setup
            # For connector usage, we might not want to contribute memory (global_segment_size=0)
            # unless specified.
            # But MooncakeStoreConfig loads global_segment_size from env.
            
            ret_code = self.store.setup(
                self.config.local_hostname,
                self.config.metadata_server,
                self.config.global_segment_size,
                local_buffer_size,
                self.config.protocol,
                self.config.device_name,
                self.config.master_server_address,
            )

        if ret_code:
            raise RuntimeError(f"Failed to setup Mooncake store, error code: {ret_code}")

        self.s, self.d = create_serde("safe")

    def get(self, key: str) -> Optional[torch.Tensor]:
        # Mooncake get returns bytes
        # We need to allocate a buffer to read into?
        # Wait, MooncakeDistributedStore.get(key) returns bytes?
        # In MooncakeStore.warmup:
        # assert self.store.get(warmup_key) == warmup_value
        # So it seems self.store.get(key) returns bytes.
        
        try:
            val = self.store.get(key)
        except Exception as e:
            logger.error(f"Error getting key {key} from Mooncake: {e}")
            return None

        if not val:
            # Mooncake might return empty bytes or None if not found?
            # Need to verify behavior. 
            # In C++ binding, it usually returns empty string or throws.
            # Based on warmup, it returns the value.
            return None

        return self.d.from_bytes(val)

    def getstr(self, key: str) -> Optional[str]:
        try:
            val = self.store.get(key)
        except Exception as e:
            logger.error(f"Error getting key {key} from Mooncake: {e}")
            return None
            
        if not val:
            return None
            
        return val.decode("utf-8")

    def set(self, key: str, tensor: torch.Tensor) -> None:
        assert tensor is not None
        val_bytes = self.s.to_bytes(tensor)
        ret = self.store.put(key, val_bytes)
        if ret != 0:
            raise RuntimeError(f"Failed to set key {key} in Mooncake, error code: {ret}")

    def setstr(self, key: str, obj: str) -> None:
        ret = self.store.put(key, obj.encode("utf-8"))
        if ret != 0:
            raise RuntimeError(f"Failed to set key {key} in Mooncake, error code: {ret}")

    def list(self, prefix: str) -> List[str]:
        # Mooncake does not support listing keys efficiently yet.
        raise NotImplementedError("MooncakeConnector does not support listing keys.")

    def weight_iterator(
        self, rank: int = 0
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        raise NotImplementedError("MooncakeConnector does not support weight iteration.")

    def pull_files(
        self,
        allow_pattern: Optional[List[str]] = None,
        ignore_pattern: Optional[List[str]] = None,
    ) -> None:
        raise NotImplementedError("MooncakeConnector does not support pulling files.")

    def close(self):
        # MooncakeDistributedStore destructor handles cleanup
        pass
