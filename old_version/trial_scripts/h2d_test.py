import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Sampler
import time
import threading
import queue
import multiprocessing as mp

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.h2d import CudaPrefetcher
from utils.async_dataloader import AsyncDataLoader


# ── 1. SamplerWorkerLoader ─────────────────────────────────────
class QueueSampler(Sampler):
    """Lightweight sampler that reads pre-computed index batches from a queue."""

    def __init__(self, index_queue: mp.Queue, n_batches: int):
        self.index_queue = index_queue
        self.n_batches   = n_batches

    def __iter__(self):
        while True:
            indices = self.index_queue.get()
            if indices is None:
                break
            yield indices

    def __len__(self):
        return self.n_batches


class SamplerWorkerLoader:
    """
    Runs the heavy sampler in a separate non-daemon process,
    feeds index batches to a standard DataLoader via QueueSampler.
    Avoids the daemon-child conflict entirely, while still allowing
    DataLoader to use num_workers for __getitem__ + collate.
    """

    def __init__(self, dataset, heavy_sampler, dataloader_kwargs: dict):
        self.dataset            = dataset
        self.heavy_sampler      = heavy_sampler
        self.dataloader_kwargs  = dataloader_kwargs

    @staticmethod
    def _sampler_worker(sampler, index_queue: mp.Queue):
        for batch_indices in sampler:
            index_queue.put(batch_indices)
        index_queue.put(None)  # sentinel: signal end of epoch

    def __iter__(self):
        index_queue   = mp.Queue(maxsize=8)
        sampler_proc  = mp.Process(
            target=SamplerWorkerLoader._sampler_worker,
            args=(self.heavy_sampler, index_queue),
            daemon=False,  # not daemon: allowed to have child processes
        )
        sampler_proc.start()

        queue_sampler = QueueSampler(index_queue, len(self.heavy_sampler))
        loader        = DataLoader(
            self.dataset,
            batch_sampler=queue_sampler,
            **self.dataloader_kwargs,
        )

        try:
            yield from loader
        finally:
            sampler_proc.join()

    def __len__(self):
        return len(self.heavy_sampler)


# ── 2. Heavy sampler ───────────────────────────────────────────
class HeavySampler(Sampler):
    """
    Simulates a computationally expensive sampler, e.g. one that
    computes per-sample importance weights or does curriculum sorting.
    Each call to __iter__ re-scores all samples before yielding indices,
    which runs in the main thread and blocks batch delivery.
    """

    def __init__(self, dataset_size: int, batch_size: int, sleep_per_batch: float = 0.01):
        self.dataset_size    = dataset_size
        self.batch_size      = batch_size
        self.sleep_per_batch = sleep_per_batch
        self.n_batches       = dataset_size // batch_size

    def __iter__(self):
        indices = torch.randperm(self.dataset_size).tolist()
        for i in range(self.n_batches):
            time.sleep(self.sleep_per_batch)
            start = i * self.batch_size
            yield indices[start : start + self.batch_size]

    def __len__(self):
        return self.n_batches


# ── 3. Collate ─────────────────────────────────────────────────
def collate_fn(batch):
    data   = torch.stack([b[0] for b in batch])
    target = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return {'data': data, 'target': target}


# ── 4. Model ───────────────────────────────────────────────────
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


# ── 5. Run one epoch ───────────────────────────────────────────
MODE_NORMAL      = "Normal              "
MODE_PREFETCH    = "CudaPrefetcher      "
MODE_ASYNC       = "Async+CudaPrefetcher"
MODE_SAMPLER_MP  = "SamplerMP+Prefetcher"


def run_epoch(loader, model, optimizer, criterion, device, mode,
              dataset=None, sampler=None, dataloader_kwargs=None):
    model.train()
    total_loss   = 0
    async_loader = None

    if mode == MODE_NORMAL:
        data_iter = loader

    elif mode == MODE_PREFETCH:
        data_iter = CudaPrefetcher(loader, device=device,
                                   cuda_keys=['data', 'target'])

    elif mode == MODE_ASYNC:
        async_loader = AsyncDataLoader(loader, buffer_size=4)
        data_iter    = CudaPrefetcher(async_loader, device=device,
                                      cuda_keys=['data', 'target'])

    elif mode == MODE_SAMPLER_MP:
        # SamplerWorkerLoader runs heavy sampler in a separate process
        # then wraps result with CudaPrefetcher for H2D overlap
        sw_loader = SamplerWorkerLoader(dataset, sampler, dataloader_kwargs)
        data_iter = CudaPrefetcher(sw_loader, device=device,
                                   cuda_keys=['data', 'target'])

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    try:
        for batch in data_iter:
            if mode == MODE_NORMAL:
                data   = batch['data'].to(device)
                target = batch['target'].to(device)
            else:
                data   = batch['data']
                target = batch['target']

            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    finally:
        if async_loader is not None:
            async_loader.stop()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return total_loss / len(loader), elapsed


# ── 6. Benchmark ───────────────────────────────────────────────
def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("No GPU found, exiting")
        return

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True,
                             transform=transform)

    sampler = HeavySampler(len(dataset), batch_size=256, sleep_per_batch=0.01)
    loader  = DataLoader(dataset, batch_sampler=sampler,
                         collate_fn=collate_fn,
                         num_workers=1, pin_memory=True, prefetch_factor=8)

    # kwargs passed to DataLoader inside SamplerWorkerLoader
    dataloader_kwargs = dict(collate_fn=collate_fn, num_workers=4, pin_memory=True)

    EPOCHS = 5
    # modes  = [MODE_NORMAL, MODE_PREFETCH, MODE_ASYNC, MODE_SAMPLER_MP]
    modes  = [MODE_ASYNC]

    for mode in modes:
        model     = SimpleNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # warmup epoch, not counted
        run_epoch(loader, model, optimizer, criterion, device, mode,
                  dataset=dataset, sampler=sampler,
                  dataloader_kwargs=dataloader_kwargs)

        times = []
        for epoch in range(EPOCHS):
            loss, t = run_epoch(loader, model, optimizer, criterion, device, mode,
                                dataset=dataset, sampler=sampler,
                                dataloader_kwargs=dataloader_kwargs)
            times.append(t)
            print(f"{mode} epoch {epoch+1} | loss: {loss:.4f} | time: {t:.3f}s")

        avg = sum(times) / len(times)
        print(f"  -> avg time: {avg:.3f}s\n")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    benchmark()
