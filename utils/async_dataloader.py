import threading
import queue

class AsyncDataLoader:
    """
    Wraps a DataLoader with a background prefetch thread.
    The background thread continuously pulls batches from the
    underlying loader and puts them into a bounded queue,
    overlapping sampler + collate with the main thread's work.
    """

    def __init__(self, loader, buffer_size: int = 4):
        self.loader      = loader
        self.buffer_size = buffer_size
        self._queue      = None
        self._thread     = None
        self._stop_event = None

    def _producer(self, stop_event: threading.Event):
        try:
            for batch in self.loader:
                if stop_event.is_set():
                    break
                self._queue.put(batch)
        except Exception as e:
            self._queue.put(e)
        finally:
            self._queue.put(StopIteration())  # instance, not the class itself

    def __iter__(self):
        self._queue      = queue.Queue(maxsize=self.buffer_size)
        self._stop_event = threading.Event()
        self._thread     = threading.Thread(
            target=self._producer,
            args=(self._stop_event,),
            daemon=True,
        )
        self._thread.start()
        return self

    def __next__(self):
        item = self._queue.get()
        if isinstance(item, StopIteration):  # check instance, not class
            self._thread.join()
            raise StopIteration
        if isinstance(item, Exception):
            self._thread.join()
            raise item
        return item

    def __len__(self):
        return len(self.loader)

    def stop(self):
        """Call if you break out of the loop early to clean up the background thread."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
