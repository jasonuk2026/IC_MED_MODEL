import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# ── 1. Prefetcher ──────────────────────────────────────────────
class CudaPrefetcher:
    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(loader)
        self.stream = torch.cuda.Stream()
        self._preload()

    def _preload(self):
        try:
            data, target = next(self.iter)
        except StopIteration:
            self.next_data = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_data   = data.cuda(non_blocking=True)
            self.next_target = target.cuda(non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data, target = self.next_data, self.next_target
        if data is None:
            raise StopIteration
        data.record_stream(torch.cuda.current_stream())
        target.record_stream(torch.cuda.current_stream())
        self._preload()
        return data, target

    def __len__(self):
        return len(self.loader)


# ── 2. 模型 ────────────────────────────────────────────────────
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


# ── 3. 训练一个 epoch，返回耗时 ────────────────────────────────
def run_epoch(loader, model, optimizer, criterion, device, use_prefetcher):
    model.train()
    total_loss = 0

    data_iter = CudaPrefetcher(loader) if use_prefetcher else loader

    # 预热：让 CUDA 初始化完成，不计入计时
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for data, target in data_iter:
        if not use_prefetcher:
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 等 GPU 上所有操作真正完成再停表
    # 不加这行的话 GPU 异步执行，计时会偏小且不准确
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return total_loss / len(loader), elapsed


# ── 4. 对比实验 ────────────────────────────────────────────────
def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("没有 GPU，Prefetcher 无效，退出")
        return

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader  = DataLoader(dataset, batch_size=256, shuffle=True,
                         num_workers=4, pin_memory=True)

    EPOCHS = 5

    for use_prefetcher in [False, True]:
        # 每次对比都重置模型和优化器，保证公平
        model     = SimpleNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # 第一个 epoch 做 CUDA 预热，不统计
        run_epoch(loader, model, optimizer, criterion, device, use_prefetcher)

        times = []
        for epoch in range(EPOCHS):
            loss, t = run_epoch(loader, model, optimizer, criterion, device, use_prefetcher)
            times.append(t)
            print(f"{'Prefetcher' if use_prefetcher else 'Normal    '} "
                  f"epoch {epoch+1} | loss: {loss:.4f} | time: {t:.3f}s")

        avg = sum(times) / len(times)
        print(f"  → 平均耗时: {avg:.3f}s\n")


if __name__ == "__main__":
    benchmark()