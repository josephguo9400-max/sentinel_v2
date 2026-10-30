import threading
import time

import uvicorn

from fakes import FakeDetector
from serving import loadtest
from serving.app import create_app
from serving.config import GatewayConfig


def test_percentile_and_small_n_marks_p99_as_max():
    s = loadtest.summarize([float(x) for x in range(1, 11)])
    assert s["p50_ms"] == 5.0 and s["p99_ms"] == 10.0 and s["p99_is_max"] is True
    big = loadtest.summarize([float(x) for x in range(1, 1001)])
    assert big["p99_ms"] == 990.0 and big["max_ms"] == 1000.0 and big["p99_is_max"] is False


def test_make_text_distinct_and_sized():
    a, b = loadtest.make_text(256, 1), loadtest.make_text(256, 2)
    assert a != b and abs(len(a.split()) - int(256 / loadtest.TOKENS_PER_WORD)) <= 1


def test_loadtest_against_live_fake_server(tmp_path):
    app = create_app(GatewayConfig(verify_model_sha256=False), FakeDetector())
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    out = tmp_path / "r.json"
    try:
        code = loadtest.main(["--n", "30", "--concurrency", "3", "--url", f"http://127.0.0.1:{port}",
                              "--lengths", "16,700", "--out", str(out), "--warmup", "1"])
    finally:
        server.should_exit = True
        thread.join(5)
    import json
    r = json.loads(out.read_text())
    assert code == 0 and r["errors"]["total"] == 0 and r["load"]["n"] == 30
    assert r["latency_ok_requests"]["count"] == 30 and r["latency_ok_requests"]["p99_is_max"] is True
    assert r["hardware"]["logical_cores"] and r["throughput_rps"] > 0
