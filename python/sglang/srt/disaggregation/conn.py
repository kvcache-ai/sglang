from __future__ import annotations

import logging
from enum import Enum
from typing import Optional, Dict, Tuple
from sglang.srt.disaggregation.transfer_engine.mooncake.mooncake import MooncakeTransferEngine
from functools import cache
import threading

import zmq

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class KVArgs:
    engine_rank: int
    kv_data_ptrs: list[int]
    kv_data_lens: list[int]
    kv_item_lens: list[int]
    aux_data_ptrs: list[int]
    aux_data_lens: list[int]
    aux_item_lens: list[int]
    ib_device: str

RequestPoolType = Dict[int, Tuple[npt.NDArray[np.int32], Optional[int]]]
KVSENDER_POLLING_PORT = 17788
KVRECIVER_POLLING_PORT = 17789

class KVManager:
    def __init__(self, args: KVArgs):
        self.engine = MooncakeTransferEngine()
        self.kv_args = args
        self.engine.register(self.kv_args)
        self.request_pool: RequestPoolType
        self.server_scoekt = zmq.Context().socket(zmq.PULL)
        self.server_scoekt.bind("tcp://*:" + KVSENDER_POLLING_PORT)

        def poll_thread():
            while True:
                (endpoint, bootstrap_room, dst_prts,
                dst_kv_indices, dst_aux_ptrs, dst_aux_index) = self.server_scoekt.recv_multipart()
                self.send_kvcache(endpoint, bootstrap_room, dst_ptrs, dst_kv_indices)
                self.send_aux(endpoint, bootstrap_room, dst_aux_ptrs, dst_aux_index)
                self.request_pool.pop(bootstrap_room)
                self._connect("tcp://" + endpoint + ":" + str(KVRECIVER_POLLING_PORT)).send_string("Done")

        threading.Thread(target=poll_thread).start()

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(self.connect_address(endpoint))
        return socket

    def send_kvcache(self, endpoint: str, bootstrap_room: int, dst_ptrs: list[int], dst_kv_indices: list[int]):
        prefill_indices, _ = self.request_pool[bootstrap_room]
        layer_num = int(len(self.kv_args.kv_data_ptrs) / 2)
        for layer_id in range(layer_num):
            prefill_key_layer_ptr = self.kv_args.kv_data_ptrs[layer_id]
            key_item_len = self.kv_args.kv_item_lens[layer_id]
            prefill_value_layer_ptr = self.kv_args.kv_data_ptrs[layer_num + layer_id]
            value_item_len = self.kv_args.kv_item_lens[layer_num + layer_id]

            decode_key_layer_ptr = dst_prts[layer_id]
            decode_value_layer_ptr = dst_prts[layer_num + layer_id]
            for prefill_index, decode_index in zip(prefill_indices, dst_kv_indices):
                prefill_key_addr = prefill_key_layer_ptr + prefill_index * key_item_len
                decode_key_addr = decode_key_layer_ptr + decode_index * key_item_len
                self.engine.transfer_sync(endpoint, prefill_key_addr, decode_key_addr, key_item_len)

                prefill_value_addr = prefill_value_layer_ptr + prefill_index * value_item_len
                decode_value_addr = decode_key_layer_ptr + decode_index * value_item_len
                self.engine.transfer_sync(endpoint, prefill_value_addr, decode_value_addr, value_item_len)

    def send_aux(self, endpoint: str, bootstrap_room: int, dst_aux_ptrs: list[int], dst_aux_index: int):
        _, prefill_aux_index = self.request_pool[bootstrap_room]
        aux_item_len = self.kv_args.aux_data_lens[0]
        prefill_aux_addr = self.kv_args.aux_data_ptrs[0] + prefill_aux_index * aux_item_len
        decode_aux_addr = dst_aux_ptrs[0] + dst_aux_index * aux_item_len
        self.engine.transfer_sync(endpoint, prefill_aux_addr, decode_aux_addr, aux_item_len)


    def enqueue_request(self, bootstrap_room: int,
                        kv_indices: npt.NDArray[np.int32],
                        aux_index: Optional[int]):
        self.request_pool[bootstrap_room] = (kv_indices, aux_index)

    def has_sent(self, bootstrap_room: int):
        if bootstrap_room in self.request_pool:
            return False
        return True

class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


class KVSender:
    def __init__(self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: int):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.aux_index = None
        self.has_sent = False

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.aux_index = aux_index
        self.num_kv_indices = num_kv_indices

    def send(self, kv_indices: npt.NDArray[np.int32]):
        self.kv_mgr.enqueue_request(self.bootstrap_room, kv_indices, self.aux_index)

    def poll(self) -> KVPoll:
        if self.has_sent is False:
            # Assume handshake completed instantly
            if self.kv_mgr.has_sent(self.bootstrap_room):
                self.has_sent = True
            return KVPoll.WaitingForInput
        else:
            # Assume transfer completed instantly
            return KVPoll.Success

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class KVReceiver:
    def __init__(
        self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: Optional[int] = None
    ):
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.prefill_server_url = bootstrap_addr.split(":")[0] + ":" + str(KVSENDER_POLLING_PORT)
        self.decode_url = ""
        self.kv_mgr = mgr
        self.has_init = False
        self.server_scoekt = zmq.Context().socket(zmq.PULL)
        self.server_scoekt.bind("tcp://*:" + KVRECIVER_POLLING_PORT)

        def poll_thread():
            while True:
                ret = self.server_scoekt.recv_string()
                if ret == "Done":
                    self.has_init = True

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(self.connect_address(endpoint))
        return socket

    def init(self, kv_indices: npt.NDArray[np.int32], aux_index: Optional[int] = None):
        self._connect("tcp://"+self.prefill_server_url).send_multipart(
            [
                self.decode_url,
                str(self.bootstrap_room),
                self.kv_mgr.kv_data_ptrs,
                kv_indices,
                self.kv_mgr.aux_data_ptrs,
                aux_index,
            ]
        )
        self.aux_index = aux_index
        self.has_init = False

    def poll(self) -> KVPoll:
        if self.has_init is False:
            # Assume handshake completed instantly
            return KVPoll.WaitingForInput
        else:
            # Assume transfer completed instantly
            return KVPoll.Success

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")


class KVBootstrapServer:
    def __init__(self, port: int): ...

    def poll(self) -> KVPoll: ...
