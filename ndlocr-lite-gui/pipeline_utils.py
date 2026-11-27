"""Three-stage pipeline helpers (prefetch/preprocess + bounded async writer) for GUI OCR."""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import queue
import threading
from typing import Callable, Optional, Sequence


_PREFETCH_DONE = object()
_PREFETCH_ERROR = object()


def default_writer_workers(cpu_count: Optional[int] = None) -> int:
    """Return bounded worker count for GUI Stage 3 CPU post-processing and I/O."""
    raw = os.cpu_count() if cpu_count is None else cpu_count
    try:
        cpus = int(raw or 2)
    except (TypeError, ValueError):
        cpus = 2
    return max(2, min(6, max(2, cpus // 2)))


def default_max_pending_writes(writer_workers: int) -> int:
    """Return bounded in-flight write future limit to cap peak page image RAM."""
    return min(8, max(2, int(writer_workers)) + 2)


def preprocess_detector_input(detector, img):
    """Run DEIM preprocessing ahead of Stage 2 when supported by ``detector``."""
    if detector is not None and hasattr(detector, 'preprocess_with_meta'):
        return detector.preprocess_with_meta(img)
    return None


class PdfPagePrefetcher:
    """Prefetch PDF pages via one reused PDFium document and parallel DEIM preprocessing."""

    def __init__(
        self,
        pdf_path: str | Path,
        page_count: int,
        render_scale: float,
        detector,
        *,
        prefetch_size: int = 3,
        preprocess_workers: int = 2,
        cancel_event: Optional[threading.Event] = None,
        pdfium_module=None,
        np_module=None,
    ) -> None:
        self._pdf_path = str(pdf_path)
        self.page_count = max(0, int(page_count))
        self._render_scale = float(render_scale)
        self._detector = detector
        self._prefetch_size = max(1, int(prefetch_size))
        self._preprocess_workers = max(1, int(preprocess_workers))
        self._cancel_event = cancel_event
        self._pdfium_module = pdfium_module
        self._np_module = np_module

        self._stop_event = threading.Event()
        self._q: queue.Queue = queue.Queue(maxsize=self._prefetch_size)
        self._closed = False
        self._exhausted = self.page_count == 0
        self._preprocess_pool: Optional[ThreadPoolExecutor] = None
        self._worker: Optional[threading.Thread] = None

        if self.page_count > 0:
            self._preprocess_pool = ThreadPoolExecutor(
                max_workers=self._preprocess_workers,
                thread_name_prefix='gui-pdf-pre',
            )
            self._worker = threading.Thread(
                target=self._render_worker,
                name='gui-pdf-render',
                daemon=True,
            )
            self._worker.start()

    def _is_cancelled(self) -> bool:
        return self._stop_event.is_set() or (
            self._cancel_event is not None and self._cancel_event.is_set()
        )

    def _put_stop_aware(self, item, *, check_cancel: bool = True) -> bool:
        while not self._stop_event.is_set():
            if check_cancel and self._cancel_event is not None and self._cancel_event.is_set():
                return False
            try:
                self._q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _render_worker(self) -> None:
        doc = None
        stream = None
        try:
            pdfium = self._pdfium_module
            if pdfium is None:
                import pypdfium2 as pdfium
            np_mod = self._np_module
            if np_mod is None:
                import numpy as np_mod

            stream = Path(self._pdf_path).open('rb')
            doc = pdfium.PdfDocument(stream)

            for page_index in range(self.page_count):
                if self._is_cancelled():
                    break
                rendered_pages = doc.render(
                    pdfium.PdfBitmap.to_pil,
                    page_indices=[page_index],
                    scale=self._render_scale,
                )
                pil_image = next(iter(rendered_pages))
                try:
                    rgb_pil = pil_image.convert('RGB')
                    try:
                        img = np_mod.array(rgb_pil, copy=True)
                    finally:
                        if rgb_pil is not pil_image and callable(getattr(rgb_pil, 'close', None)):
                            rgb_pil.close()
                finally:
                    if callable(getattr(pil_image, 'close', None)):
                        pil_image.close()

                if self._is_cancelled():
                    break

                assert self._preprocess_pool is not None
                try:
                    pre_future = self._preprocess_pool.submit(
                        preprocess_detector_input,
                        self._detector,
                        img,
                    )
                except RuntimeError:
                    break

                if not self._put_stop_aware((page_index, img, pre_future), check_cancel=True):
                    pre_future.cancel()
                    break
            self._put_stop_aware((_PREFETCH_DONE, None, None), check_cancel=False)
        except BaseException as exc:
            if not self._stop_event.is_set():
                self._put_stop_aware((_PREFETCH_ERROR, exc, None), check_cancel=False)
        finally:
            try:
                if doc is not None:
                    doc.close()
            finally:
                if stream is not None:
                    stream.close()

    def _get_next_sync(self):
        if self._closed or self._exhausted:
            return None
        tag, payload, pre_future = self._q.get()
        if tag is _PREFETCH_DONE:
            self._exhausted = True
            return None
        if tag is _PREFETCH_ERROR:
            self._exhausted = True
            raise payload
        det_pre = pre_future.result() if pre_future is not None else None
        return tag, payload, det_pre

    async def next_page(self):
        """Return ``(page_index, img, det_pre)`` for the next page, or ``None``."""
        if self._closed or self._exhausted:
            return None
        return await asyncio.to_thread(self._get_next_sync)

    def close(self) -> None:
        """Stop prefetching, drain queued items, and release PDFium/thread resources."""
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        while True:
            try:
                _, _, extra = self._q.get_nowait()
                if extra is not None and callable(getattr(extra, 'cancel', None)):
                    extra.cancel()
            except queue.Empty:
                break
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=3.0)
        if self._preprocess_pool is not None:
            wait = not (self._worker is not None and self._worker.is_alive())
            self._preprocess_pool.shutdown(wait=wait, cancel_futures=True)
            self._preprocess_pool = None


class ImagePrefetcher:
    """Bounded sliding-window prefetcher that loads images and runs DEIM preprocessing."""

    def __init__(
        self,
        paths: Sequence[str],
        load_image_fn: Callable[[str], object],
        detector,
        *,
        prefetch_size: int = 3,
        workers: int = 2,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self._paths = list(paths)
        self._load_image_fn = load_image_fn
        self._detector = detector
        self._prefetch_size = max(1, int(prefetch_size))
        self._cancel_event = cancel_event
        self._executor = (
            ThreadPoolExecutor(
                max_workers=max(1, int(workers)),
                thread_name_prefix='gui-img-prefetch',
            )
            if self._paths
            else None
        )
        self._pending: deque = deque()
        self._next_submit = 0
        self._closed = False
        self._fill()

    def _produce_one(self, path: str):
        img = self._load_image_fn(path)
        det_pre = preprocess_detector_input(self._detector, img)
        return img, det_pre

    def _fill(self) -> None:
        if self._closed or self._executor is None:
            return
        while (
            self._next_submit < len(self._paths)
            and len(self._pending) < self._prefetch_size
        ):
            if self._cancel_event is not None and self._cancel_event.is_set():
                break
            idx = self._next_submit
            path = self._paths[idx]
            fut = self._executor.submit(self._produce_one, path)
            self._pending.append((idx, path, fut))
            self._next_submit += 1

    async def next_item(self):
        """Return ``(idx, path, img, det_pre)`` in order, or ``None`` when finished."""
        if self._closed:
            return None
        if self._cancel_event is not None and self._cancel_event.is_set():
            return None
        if not self._pending:
            self._fill()
        if not self._pending:
            return None
        idx, path, fut = self._pending.popleft()
        self._fill()
        img, det_pre = await asyncio.wrap_future(fut)
        return idx, path, img, det_pre

    def close(self) -> None:
        """Cancel queued futures and shut down the prefetch thread pool."""
        if self._closed:
            return
        self._closed = True
        while self._pending:
            _, _, fut = self._pending.popleft()
            fut.cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None


class BoundedAsyncWriter:
    """Async writer pool with backpressure and FIFO completion callbacks."""

    def __init__(
        self,
        max_workers: Optional[int] = None,
        max_pending: Optional[int] = None,
        *,
        thread_name_prefix: str = 'gui-writer',
    ) -> None:
        workers = default_writer_workers() if max_workers is None else max(1, int(max_workers))
        self.max_workers = workers
        self.max_pending = (
            default_max_pending_writes(workers)
            if max_pending is None
            else max(1, int(max_pending))
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._pending: deque = deque()
        self._closed = False

    def flush_completed(self) -> None:
        """Consume already-completed prefix futures in FIFO submission order."""
        while self._pending and self._pending[0][0].done():
            fut, on_complete = self._pending.popleft()
            result = fut.result()
            if on_complete is not None:
                on_complete(result)

    async def submit(self, fn, *args, on_complete=None, **kwargs) -> None:
        """Enqueue a write task, awaiting the oldest task first if at capacity."""
        if self._closed:
            raise RuntimeError('BoundedAsyncWriter is already closed')
        self.flush_completed()
        while len(self._pending) >= self.max_pending:
            fut, cb = self._pending.popleft()
            result = await asyncio.wrap_future(fut)
            if cb is not None:
                cb(result)
        fut = self._executor.submit(fn, *args, **kwargs)
        self._pending.append((fut, on_complete))

    async def drain(self, suppress_errors: bool = False) -> None:
        """Wait for all pending tasks in FIFO order before raising any error."""
        first_error: Optional[BaseException] = None
        while self._pending:
            fut, cb = self._pending.popleft()
            try:
                result = await asyncio.wrap_future(fut)
                if cb is not None:
                    cb(result)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None and not suppress_errors:
            raise first_error

    async def close(self, suppress_errors: bool = False) -> None:
        """Drain remaining write tasks and shut down the thread pool."""
        try:
            await self.drain(suppress_errors=suppress_errors)
        finally:
            self._closed = True
            self._executor.shutdown(wait=True)
