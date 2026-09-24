"""`_port_holder` must see a listener on the WILDCARD address.

AUDIT F16-1: the probe is a bind to 127.0.0.1:port. On Windows that bind
succeeds while another process already listens on 0.0.0.0:port (the two
addresses only overlap when SO_EXCLUSIVEADDRUSE is set), and the psutil fallback
only ran after a bind failure — so it never ran in the one case it exists for.

Everything here is in-process: a wildcard listener bound to an ephemeral port is
created, the real `_port_holder` is called, and the socket is closed again. No
external process is touched and no traffic is sent.
"""
import os
import socket

import rigma.cli as cli


def test_port_holder_sees_a_wildcard_listener():
    srv = socket.socket()
    srv.bind(("0.0.0.0", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        holder = cli._port_holder(port)
    finally:
        srv.close()
    assert holder != "", "a wildcard listener must not read as a free port"
    assert f"pid {os.getpid()}" in holder


def test_port_holder_reports_a_free_port():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    assert cli._port_holder(port) == ""


def test_port_holder_sees_a_loopback_listener():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        holder = cli._port_holder(port)
    finally:
        srv.close()
    assert f"pid {os.getpid()}" in holder


def test_port_holder_names_a_known_holder(monkeypatch):
    """The reported pid is looked up for a name; a lookup failure must not turn
    a held port into a free one."""
    import psutil
    monkeypatch.setattr(cli, "_listening_pid", lambda port: 4242)

    def no_name(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", no_name)
    assert "pid 4242" in cli._port_holder(65000)
