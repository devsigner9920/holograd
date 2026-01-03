import pickle
import struct
from dataclasses import dataclass
from typing import Any, Optional
import zmq

MSG_TASK = b"TASK____"
MSG_PROOF = b"PROOF___"
MSG_PARAMS = b"PARAMS__"
MSG_CODEBOOK = b"CODEBOOK"
MSG_SHUTDOWN = b"SHUTDOWN"
MSG_READY = b"READY___"
MSG_ACK = b"ACK_____"

MSG_HEADER_SIZE = 8


@dataclass
class Message:
    msg_type: bytes
    payload: Any


def serialize(msg: Message) -> bytes:
    payload_bytes = pickle.dumps(msg.payload)
    header = msg.msg_type[:MSG_HEADER_SIZE].ljust(MSG_HEADER_SIZE, b"_")
    return header + struct.pack(">I", len(payload_bytes)) + payload_bytes


def deserialize(data: bytes) -> Message:
    msg_type = data[:MSG_HEADER_SIZE].rstrip(b"_")
    payload_len = struct.unpack(">I", data[MSG_HEADER_SIZE : MSG_HEADER_SIZE + 4])[0]
    payload_bytes = data[MSG_HEADER_SIZE + 4 : MSG_HEADER_SIZE + 4 + payload_len]
    payload = pickle.loads(payload_bytes)
    return Message(msg_type=msg_type, payload=payload)


class CoordinatorServer:
    def __init__(self, port: int = 5555):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://*:{port}")
        self.workers: dict[bytes, bool] = {}

    def wait_for_workers(self, num_workers: int, timeout: int = 300) -> list[bytes]:
        self.socket.setsockopt(zmq.RCVTIMEO, timeout * 1000)

        while len(self.workers) < num_workers:
            try:
                frames = self.socket.recv_multipart()
                identity = frames[0]
                data = frames[-1]
                msg = deserialize(data)

                if msg.msg_type == b"READY":
                    self.workers[identity] = True
                    print(f"[Coordinator] Worker {len(self.workers)}/{num_workers} connected")
                    self.socket.send_multipart([identity, serialize(Message(MSG_ACK, None))])
            except zmq.Again:
                raise TimeoutError(
                    f"Timeout waiting for workers. Got {len(self.workers)}/{num_workers}"
                )

        return list(self.workers.keys())

    def broadcast_params(self, worker_ids: list[bytes], params: Any) -> None:
        msg = serialize(Message(MSG_PARAMS, params))
        for wid in worker_ids:
            self.socket.send_multipart([wid, msg])

    def broadcast_codebook(self, worker_ids: list[bytes], codebook_data: Any) -> None:
        msg = serialize(Message(MSG_CODEBOOK, codebook_data))
        for wid in worker_ids:
            self.socket.send_multipart([wid, msg])

    def send_tasks(self, worker_tasks: list[tuple[bytes, Any]]) -> None:
        for wid, task in worker_tasks:
            msg = serialize(Message(MSG_TASK, task))
            self.socket.send_multipart([wid, msg])

    def collect_proofs(self, num_proofs: int, timeout: int = 60) -> list[Any]:
        self.socket.setsockopt(zmq.RCVTIMEO, timeout * 1000)
        proofs = []

        while len(proofs) < num_proofs:
            try:
                frames = self.socket.recv_multipart()
                data = frames[-1]
                msg = deserialize(data)

                if msg.msg_type == b"PROOF":
                    proofs.append(msg.payload)
            except zmq.Again:
                print(f"[Coordinator] Timeout collecting proofs. Got {len(proofs)}/{num_proofs}")
                break

        return proofs

    def shutdown_workers(self, worker_ids: list[bytes]) -> None:
        msg = serialize(Message(MSG_SHUTDOWN, None))
        for wid in worker_ids:
            self.socket.send_multipart([wid, msg])

    def close(self) -> None:
        self.socket.close()
        self.context.term()


class WorkerClient:
    def __init__(self, coordinator_host: str, coordinator_port: int = 5555, worker_id: int = 0):
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.worker_id = worker_id

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.IDENTITY, f"worker-{worker_id}".encode())
        self.socket.connect(f"tcp://{coordinator_host}:{coordinator_port}")

    def register(self) -> bool:
        msg = serialize(Message(MSG_READY, {"worker_id": self.worker_id}))
        self.socket.send(msg)

        self.socket.setsockopt(zmq.RCVTIMEO, 30000)
        try:
            data = self.socket.recv()
            response = deserialize(data)
            return response.msg_type == b"ACK"
        except zmq.Again:
            return False

    def receive(self, timeout: Optional[int] = None) -> Optional[Message]:
        if timeout:
            self.socket.setsockopt(zmq.RCVTIMEO, timeout * 1000)
        else:
            self.socket.setsockopt(zmq.RCVTIMEO, -1)

        try:
            data = self.socket.recv()
            return deserialize(data)
        except zmq.Again:
            return None

    def send_proof(self, proof: Any) -> None:
        msg = serialize(Message(MSG_PROOF, proof))
        self.socket.send(msg)

    def close(self) -> None:
        self.socket.close()
        self.context.term()
