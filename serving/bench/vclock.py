"""A virtual-time asyncio event loop for the streaming harness and its tests.

`loop.time()` returns a virtual clock that jumps to the next scheduled timer whenever no callback is ready, so
`asyncio.sleep` costs no wall time and every timing is exactly reproducible. Only coroutines that wait on timers
may run on it: a thread pool, a subprocess or socket I/O would see the clock jump past them. The streaming harness
therefore runs the real classifier on a normal loop in another thread and charges its *measured* latency to the
virtual clock (serving/bench/streaming.py).

Relies on two private attributes of CPython's BaseEventLoop (`_ready`, `_scheduled`); measurement and test code
only, never imported by the gateway.
"""
from __future__ import annotations

import asyncio


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    def __init__(self):
        super().__init__()
        self._vt = 0.0

    def time(self) -> float:  # noqa: D401 - asyncio API
        return self._vt

    def _run_once(self):
        if not self._ready and self._scheduled:
            live = [h._when for h in self._scheduled if not h._cancelled]
            if live and min(live) > self._vt:
                self._vt = min(live)
        super()._run_once()


def run_virtual(coro):
    """Run a coroutine to completion on a fresh VirtualTimeLoop."""
    loop = VirtualTimeLoop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
