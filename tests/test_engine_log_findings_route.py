"""A17/S2 is wired to the EXISTING `/api/server/findings` surface.

The route already returns `engine_log.findings(text)`, so the load accounting
reaches the Engine panel without a second endpoint and without a `serve.py`
change. What this file pins is the behaviour the owner sees:

  * the real healthy load adds NO finding — the defect was that a constant
    `expected_splits = 1` flagged it on every launch;
  * a synthetic CPU attention fallback DOES add one;
  * a MoE that offloads experts reads "not comparable" and adds nothing.

The log lines are the same verbatim fixture lines as
`tests/test_engine_log_memory.py`, taken from `.scratch/prism-v.log`
(lines 2466 and 4485-4672) with `graph splits` varied for the synthetic cases.
"""
import pytest
from fastapi.testclient import TestClient

from rigma.serve import build_app

HEALTHY_LOAD = (
    "0.00.987.636 I print_info: n_expert              = 0\n"
    "0.01.522.670 I load_tensors: offloaded 65/65 layers to GPU\n"
    "0.01.522.675 I load_tensors:   CPU_Mapped model buffer size =   322.07 MiB\n"
    "0.01.522.676 I load_tensors:        ROCm0 model buffer size =  6539.67 MiB\n"
    "0.04.364.938 I llama_kv_cache:      ROCm0 KV buffer size =  2176.00 MiB\n"
    "0.04.369.729 I llama_memory_recurrent:      ROCm0 RS buffer size =   149.62 MiB\n"
    "0.04.417.931 I sched_reserve:      ROCm0 compute buffer size =   410.28 MiB\n"
    "0.04.417.938 I sched_reserve:  ROCm_Host compute buffer size =    84.28 MiB\n"
    "0.04.417.939 I sched_reserve: graph splits = 2\n"
)

CPU_ATTENTION_FALLBACK = HEALTHY_LOAD.replace("graph splits = 2",
                                              "graph splits = 34")

MOE_EXPERT_OFFLOAD = (
    "0.00.987.636 I print_info: n_expert              = 256\n"
    "0.00.987.637 I print_info: n_expert_used         = 8\n"
    "0.01.522.670 I load_tensors: offloaded 49/49 layers to GPU\n"
    "0.01.522.676 I load_tensors:        ROCm0 model buffer size = 18000.00 MiB\n"
    "0.01.522.677 I load_tensors:           CPU model buffer size =  2000.00 MiB\n"
    "0.04.417.931 I sched_reserve:      ROCm0 compute buffer size =   600.00 MiB\n"
    "0.04.417.939 I sched_reserve: graph splits = 98\n"
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return TestClient(build_app(upstream_port=1, default_prompt=""))


def _serve_log(tmp_path, text):
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "server-1.log").write_text(text, encoding="utf-8")


def _ids(client):
    r = client.get("/api/server/findings")
    assert r.status_code == 200
    return [f["id"] for f in r.json()["findings"]]


def test_a_healthy_load_yields_no_finding_from_the_route(tmp_path, client):
    _serve_log(tmp_path, HEALTHY_LOAD)

    assert _ids(client) == []


def test_a_cpu_attention_fallback_yields_a_finding_from_the_route(
        tmp_path, client):
    _serve_log(tmp_path, CPU_ATTENTION_FALLBACK)

    assert _ids(client) == ["plan_divergence"]


def test_a_moe_expert_offload_yields_no_finding_from_the_route(
        tmp_path, client):
    # "not comparable" is not a finding; the owner's 35B MoE must not be flagged.
    _serve_log(tmp_path, MOE_EXPERT_OFFLOAD)

    assert _ids(client) == []
