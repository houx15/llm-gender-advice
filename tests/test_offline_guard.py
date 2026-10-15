"""Offline guard: local entry points raise if offline env vars are missing or the network is touched."""
import socket
import urllib.request

import pytest

from advice_audit.local import offline_guard as og

ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
       "VLLM_NO_USAGE_STATS": "1", "HF_HOME": "/tmp/hf"}


@pytest.fixture
def clean_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    yield
    og.deactivate()


def test_missing_env_raises(clean_env):
    with pytest.raises(og.OfflineEnvError):
        og.activate()


@pytest.mark.parametrize("module", ["generate", "extract", "probes", "patching", "steering", "erasure", "readout"])
def test_entry_points_refuse_without_env(clean_env, module):
    import importlib

    mod = importlib.import_module(f"advice_audit.local.{module}")
    with pytest.raises(og.OfflineEnvError):
        mod.main(["--model", "x", "--shard", "0"])


def test_network_blocked_loopback_allowed(clean_env, monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    og.activate()
    with pytest.raises(og.NetworkAccessError):
        socket.create_connection(("huggingface.co", 443), timeout=1)
    with pytest.raises(og.NetworkAccessError):
        urllib.request.urlopen("http://example.com", timeout=1)
    s = socket.socket()
    with pytest.raises(og.NetworkAccessError):
        s.connect(("93.184.216.34", 80))
    s.close()
    # loopback stays usable (vLLM/torch IPC)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    c = socket.create_connection(srv.getsockname(), timeout=1)
    c.close()
    srv.close()
