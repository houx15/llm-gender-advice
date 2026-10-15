"""Offline guard for compute-node entry points (spec §6.1).

`activate()` must run first in every local entry point. It asserts the offline
environment variables set by scripts/slurm/common.sh and monkeypatches Python socket
connects so that ANY non-loopback network call raises immediately. Loopback and Unix
sockets stay allowed (vLLM / torch use them for intra-node IPC).
"""
from __future__ import annotations

import ipaddress
import os
import socket

REQUIRED_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "VLLM_NO_USAGE_STATS": "1",
}
REQUIRED_SET = ("HF_HOME",)


class NetworkAccessError(RuntimeError):
    """Raised when code on a compute node tries to reach the network."""


class OfflineEnvError(RuntimeError):
    """Raised when the offline environment variables are missing."""


_ORIG_CREATE_CONNECTION = socket.create_connection
_ORIG_CONNECT = socket.socket.connect
_ORIG_CONNECT_EX = socket.socket.connect_ex
_ACTIVE = False


def _is_local(address) -> bool:
    if isinstance(address, (str, bytes)):  # AF_UNIX path
        return True
    host = address[0] if isinstance(address, tuple) and address else address
    if isinstance(host, bytes):
        host = host.decode()
    if host in ("localhost", "", None):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # hostname other than localhost -> would need DNS -> treat as network


def _blocked_create_connection(address, *args, **kwargs):
    if not _is_local(address):
        raise NetworkAccessError(f"offline guard: network access attempted to {address!r}")
    return _ORIG_CREATE_CONNECTION(address, *args, **kwargs)


def _blocked_connect(self, address):
    if self.family != getattr(socket, "AF_UNIX", None) and not _is_local(address):
        raise NetworkAccessError(f"offline guard: network access attempted to {address!r}")
    return _ORIG_CONNECT(self, address)


def _blocked_connect_ex(self, address):
    if self.family != getattr(socket, "AF_UNIX", None) and not _is_local(address):
        raise NetworkAccessError(f"offline guard: network access attempted to {address!r}")
    return _ORIG_CONNECT_EX(self, address)


def check_env() -> None:
    missing = [f"{k}={v}" for k, v in REQUIRED_ENV.items() if os.environ.get(k) != v]
    missing += [k for k in REQUIRED_SET if not os.environ.get(k)]
    if missing:
        raise OfflineEnvError(
            "offline guard: required environment not set: " + ", ".join(missing)
            + " (source scripts/slurm/common.sh)"
        )


def activate() -> None:
    """Assert offline env vars and block network sockets. Idempotent."""
    global _ACTIVE
    check_env()
    socket.create_connection = _blocked_create_connection
    socket.socket.connect = _blocked_connect
    socket.socket.connect_ex = _blocked_connect_ex
    _ACTIVE = True


def deactivate() -> None:
    """Restore sockets (tests only)."""
    global _ACTIVE
    socket.create_connection = _ORIG_CREATE_CONNECTION
    socket.socket.connect = _ORIG_CONNECT
    socket.socket.connect_ex = _ORIG_CONNECT_EX
    _ACTIVE = False


def is_active() -> bool:
    return _ACTIVE
