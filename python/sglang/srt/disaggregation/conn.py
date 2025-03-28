from __future__ import annotations

import logging
from enum import Enum
from typing import Optional, Dict, Tuple
from sglang.srt.disaggregation.transfer_engine.mooncake_conn import MooncakeTransferEngine
import threading
from functools import cache

import zmq
import struct

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
        self.request_pool: RequestPoolType = {0 : (np.array([0], dtype=np.int32), None)}
        self.server_socket = zmq.Context().socket(zmq.PULL)
        self.register_to_engine()
        self.prefill_thread_started = False
        self.decode_thread_started = False

    def register_to_engine(self):
        for kv_data_ptr, kv_data_len in zip(self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens):
            self.engine.register(kv_data_ptr, kv_data_len)

        for aux_data_ptr, aux_data_len in zip(self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens):
            self.engine.register(aux_data_ptr, aux_data_len)

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(endpoint)
        return socket

    def get_kvcache(self, endpoint: str, bootstrap_room: int, prefill_ptrs: list[int], prefill_kv_indices: list[int]):
        decode_kv_indices, _ = self.request_pool[bootstrap_room]
        layer_num = int(len(self.kv_args.kv_data_ptrs) / 2)
        for layer_id in range(layer_num):
            decode_key_layer_ptr = self.kv_args.kv_data_ptrs[layer_id]
            key_item_len = self.kv_args.kv_item_lens[layer_id]
            decode_value_layer_ptr = self.kv_args.kv_data_ptrs[layer_num + layer_id]
            value_item_len = self.kv_args.kv_item_lens[layer_num + layer_id]

            prefill_key_layer_ptr = prefill_ptrs[layer_id]
            prefill_value_layer_ptr = prefill_ptrs[layer_num + layer_id]
            for prefill_index, decode_index in zip(prefill_kv_indices, decode_kv_indices):
                prefill_key_addr = prefill_key_layer_ptr + prefill_index * key_item_len
                decode_key_addr = decode_key_layer_ptr + decode_index * key_item_len
                self.engine.transfer_sync(endpoint, decode_key_addr, prefill_key_addr, key_item_len)

                prefill_value_addr = prefill_value_layer_ptr + prefill_index * value_item_len
                decode_value_addr = decode_key_layer_ptr + decode_index * value_item_len
                self.engine.transfer_sync(endpoint, decode_value_addr, prefill_value_addr, value_item_len)

    def get_aux(self, endpoint: str, bootstrap_room: int, prefill_aux_ptrs: list[int], prefill_aux_index: int):
        _, decode_aux_index = self.request_pool[bootstrap_room]
        aux_item_len = self.kv_args.aux_data_lens[0]
        decode_aux_addr = self.kv_args.aux_data_ptrs[0] + decode_aux_index * aux_item_len
        prefill_aux_addr = prefill_aux_ptrs[0] + prefill_aux_index * aux_item_len
        self.engine.transfer_sync(endpoint, decode_aux_addr, prefill_aux_addr, aux_item_len)

    def start_prefill_thread(self):
        if self.prefill_thread_started == True:
            return
        self.prefill_thread_started = True
        self.server_socket.bind("tcp://*:" + str(KVSENDER_POLLING_PORT))
        def prefill_thread():
            while True:
                decode_ip, bootstrap_room, status = self.server_socket.recv_multipart()
                if bootstrap_room.decode('ascii') == 'None':
                    continue

                bootstrap_room = int(bootstrap_room.decode('ascii'))

                if status.decode('ascii') == 'Done':
                    self.request_pool.pop(bootstrap_room)
                    continue

                decode_ip = decode_ip.decode('ascii')
                endpoint = self.engine.get_localhost()
                prefill_kv_indices, prefill_aux_index = self.request_pool[bootstrap_room]
                packed_kv_data_ptrs = b''.join(struct.pack('q', ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs)
                packed_aux_data_ptrs = b''.join(struct.pack('q', ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs)
                decode_server_url = decode_ip + ":" + str(KVRECIVER_POLLING_PORT)
                self._connect("tcp://" + decode_server_url).send_multipart(
                    [
                        endpoint.encode('ascii'),
                        str(bootstrap_room).encode('ascii'),
                        packed_kv_data_ptrs,
                        prefill_kv_indices.tobytes(),
                        packed_aux_data_ptrs,
                        str(prefill_aux_index).encode('ascii'),
                    ]
                )

        threading.Thread(target=prefill_thread).start()
        print("Prefill thread start")

    def start_decode_thread(self):
        if self.decode_thread_started == True:
            return
        self.decode_thread_started = True
        self.server_socket.bind("tcp://*:" + str(KVRECIVER_POLLING_PORT))
        def decode_thread():
            while True:
                (endpoint, bootstrap_room, prefill_ptrs,
                prefill_kv_indices, prefill_aux_ptrs, prefill_aux_index) = self.server_socket.recv_multipart()
                endpoint = endpoint.decode('ascii')
                bootstrap_room = int(bootstrap_room.decode('ascii'))
                prefill_ptrs = list(struct.unpack(f'{len(prefill_ptrs)//8}q', prefill_ptrs))
                prefill_kv_indices = np.frombuffer(prefill_kv_indices, dtype=np.int32)
                prefill_aux_ptrs = list(struct.unpack(f'{len(prefill_aux_ptrs)//8}q', prefill_aux_ptrs))
                prefill_aux_index = int(prefill_aux_index.decode('ascii'))
                self.get_kvcache(endpoint, bootstrap_room, dst_ptrs, dst_kv_indices)
                self.get_aux(endpoint, bootstrap_room, dst_aux_ptrs, dst_aux_index)
                self.request_pool.pop(bootstrap_room)
                prefill_server_url = endpoint + ":" + str(KVSENDER_POLLING_PORT)
                self._connect("tcp://"+self.prefill_server_url).send_multipart(
                    [
                        "".encode('ascii'),
                        str(self.bootstrap_room).encode('ascii'),
                        "Done".encode('ascii'),
                    ]
                )

        threading.Thread(target=decode_thread).start()
        print("Decode thread start")

    def enqueue_request(self, bootstrap_room: int,
                        kv_indices: npt.NDArray[np.int32],
                        aux_index: Optional[int]):
        self.request_pool[bootstrap_room] = (kv_indices, aux_index)

    def has_finished(self, bootstrap_room: int):
        if bootstrap_room in self.request_pool:
            return False
        return True

    def get_localhost(self):
        return self.engine.get_localhost()

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
        self.kv_mgr.start_prefill_thread()

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.aux_index = aux_index
        self.num_kv_indices = num_kv_indices

    def send(self, kv_indices: npt.NDArray[np.int32]):
        self.kv_mgr.enqueue_request(self.bootstrap_room, kv_indices, self.aux_index)

    def poll(self) -> KVPoll:
        if self.has_sent is False:
            if self.kv_mgr.has_finished(self.bootstrap_room):
                self.has_sent = True
            return KVPoll.WaitingForInput
        else:
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
        self.kv_mgr = mgr
        self.decode_ip = self.kv_mgr.get_localhost()
        self.has_init = False
        self.kv_mgr.start_decode_thread()

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(endpoint)
        return socket

    def init(self, kv_indices: npt.NDArray[np.int32], aux_index: Optional[int] = None):
        self.aux_index = aux_index
        self.kv_mgr.enqueue_request(self.bootstrap_room, kv_indices, self.aux_index)
        self._connect("tcp://" + self.prefill_server_url).send_multipart(
            [
                self.decode_ip.encode('ascii'),
                str(self.bootstrap_room).encode('ascii'),
                "Started".encode('ascii'),
            ]
        )
        self.has_init = False

    def poll(self) -> KVPoll:
        if self.has_init is False:
            if self.kv_mgr.has_finished(self.bootstrap_room):
                self.has_init = True
            return KVPoll.WaitingForInput
        else:
            return KVPoll.Success

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")


class KVBootstrapServer:
    def __init__(self, port: int): ...

    def poll(self) -> KVPoll: ...
