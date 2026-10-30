"""Async micro-batcher in front of a bounded inference worker pool.

Chunks from any number of requests are queued; a single dispatcher task takes up to `max_batch`
chunks (waiting at most `max_wait_ms` after the first one) and runs them as one `infer` call in a
worker thread. The dispatcher only takes a batch when a worker slot is free, so queued work stays in
the asyncio queue, where a request that times out can still withdraw it. Once a batch is handed to
a worker it cannot be cancelled: onnxruntime offers no way to interrupt `InferenceSession.run`.
"""
import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

InferFn = Callable[[list[list[int]]], list[float]]


class Overloaded(Exception):
    """The chunk queue is full; the caller applies its failure policy."""


class MicroBatcher:
    def __init__(self, infer: InferFn, *, max_batch: int, max_wait_ms: float, workers: int, max_queue: int,
                 on_batch: Callable[[int], None] | None = None):
        self._infer = infer
        self.max_batch = max_batch
        self.max_wait_s = max_wait_ms / 1000.0
        self.workers = workers
        self.max_queue = max_queue
        self._on_batch = on_batch
        self._queue: asyncio.Queue[tuple[list[int], asyncio.Future]] | None = None
        self._slots: asyncio.Semaphore | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._task: asyncio.Task | None = None
        self.busy_workers = 0

    async def start(self) -> None:
        self._queue = asyncio.Queue()
        self._slots = asyncio.Semaphore(self.workers)
        self._executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="sentinel-infer")
        self._task = asyncio.create_task(self._dispatch_loop(), name="sentinel-batcher")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._executor is not None:
            # Do not wait: a hung inference call would block shutdown forever.
            self._executor.shutdown(wait=False, cancel_futures=True)

    @property
    def queue_depth(self) -> int:
        return 0 if self._queue is None else self._queue.qsize()

    def submit(self, chunks: list[list[int]]) -> list[asyncio.Future]:
        """Queue all chunks of one request atomically, or raise Overloaded."""
        assert self._queue is not None, "batcher not started"
        if self._queue.qsize() + len(chunks) > self.max_queue:
            raise Overloaded(f"queue full ({self._queue.qsize()} queued, max {self.max_queue})")
        loop = asyncio.get_running_loop()
        futures = []
        for ids in chunks:
            fut = loop.create_future()
            self._queue.put_nowait((ids, fut))
            futures.append(fut)
        return futures

    async def _next_live(self, timeout: float | None) -> tuple[list[int], asyncio.Future] | None:
        """Next queued chunk whose request is still waiting; None if `timeout` expires first."""
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while True:
            if deadline is None:
                item = await self._queue.get()
            else:
                remaining = deadline - loop.time()
                try:
                    item = self._queue.get_nowait() if remaining <= 0 else await asyncio.wait_for(
                        self._queue.get(), remaining)
                except (asyncio.QueueEmpty, TimeoutError):
                    return None
            if not item[1].done():  # skip chunks of requests that already timed out
                return item

    async def _dispatch_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await self._slots.acquire()
            try:
                batch = [await self._next_live(None)]
                while len(batch) < self.max_batch:
                    item = await self._next_live(self.max_wait_s)
                    if item is None:
                        break
                    batch.append(item)
            except BaseException:
                self._slots.release()
                raise
            self.busy_workers += 1
            if self._on_batch is not None:
                self._on_batch(len(batch))
            job = loop.run_in_executor(self._executor, self._infer, [ids for ids, _ in batch])
            job.add_done_callback(lambda j, b=batch: self._finish(j, b))

    def _finish(self, job: asyncio.Future, batch: list[tuple[list[int], asyncio.Future]]) -> None:
        self.busy_workers -= 1
        self._slots.release()
        if job.cancelled():
            for _, fut in batch:
                if not fut.done():
                    fut.cancel()
            return
        exc = job.exception()
        if exc is None:
            scores = job.result()
            if len(scores) != len(batch):
                exc = RuntimeError(f"detector returned {len(scores)} scores for {len(batch)} inputs")
        for i, (_, fut) in enumerate(batch):
            if fut.done():  # its request already timed out
                continue
            if exc is not None:
                fut.set_exception(exc)
            else:
                fut.set_result(scores[i])
