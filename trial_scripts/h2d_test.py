import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Sampler
import time
import threading
import queue

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.h2d import CudaPrefetcher
from utils.async_dataloader import AsyncDataLoader


# ── 2. Heavy sampler ───────────────────────────────────────────
class HeavySampler(Sampler):
    """
    Simulates a computationally expensive sampler, e.g. one that
    computes per-sample importance weights or does curriculum sorting.
    Each call to __iter__ re-scores all samples before yielding indices,
    which runs in the main thread and blocks batch delivery.
    """

    def __init__(self, dataset_size: int, batch_size: int, sleep_per_batch: float = 0.01):
        """
        sleep_per_batch: seconds of fake computation per batch index generated.
                         0.01s * 235 batches ~ 2.35s overhead per epoch.
        """
        self.dataset_size    = dataset_size
        self.batch_size      = batch_size
        self.sleep_per_batch = sleep_per_batch
        self.n_batches       = dataset_size // batch_size

    def __iter__(self):
        indices = torch.randperm(self.dataset_size).tolist()
        for i in range(self.n_batches):
            # simulate expensive per-batch computation in main thread
            # e.g. priority queue update, loss-based reweighting
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
MODE_NORMAL   = "Normal              "
MODE_PREFETCH = "CudaPrefetcher      "
MODE_ASYNC    = "Async+CudaPrefetcher"


def run_epoch(loader, model, optimizer, criterion, device, mode):
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

    # HeavySampler runs in main thread, blocks between batches
    # 0.01s per batch * 234 batches ~ 2.3s extra overhead per epoch
    sampler = HeavySampler(len(dataset), batch_size=256, sleep_per_batch=0.01)
    loader  = DataLoader(dataset, batch_sampler=sampler,
                         collate_fn=collate_fn,
                         num_workers=4, pin_memory=True)

    EPOCHS = 5

    for mode in [MODE_NORMAL, MODE_PREFETCH, MODE_ASYNC]:
        model     = SimpleNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # warmup epoch, not counted
        run_epoch(loader, model, optimizer, criterion, device, mode)

        times = []
        for epoch in range(EPOCHS):
            loss, t = run_epoch(loader, model, optimizer, criterion, device, mode)
            times.append(t)
            print(f"{mode} epoch {epoch+1} | loss: {loss:.4f} | time: {t:.3f}s")

        avg = sum(times) / len(times)
        print(f"  -> avg time: {avg:.3f}s\n")


if __name__ == "__main__":
    benchmark()