import torch

class CudaPrefetcher:
    def __init__(self, loader, device, cuda_keys=None):
        """
        loader     : any DataLoader
        device     : target device, e.g. torch.device('cuda:0')
        cuda_keys  : keys to move to GPU
                     - None -> batch is a tensor, move the whole thing
                     - []   -> move all keys
                     - ['image', 'label'] -> move specified keys only
        """
        self.loader = loader
        self.iter = iter(loader)
        self.stream = torch.cuda.Stream(device=device)
        self.device = device
        self.cuda_keys = cuda_keys
        self._preload()

    def _to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)

        if isinstance(batch, dict):
            keys = self.cuda_keys if self.cuda_keys else batch.keys()
            return {
                k: v.to(self.device, non_blocking=True) if k in keys and isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _record_streams(self, batch):
        stream = torch.cuda.current_stream(self.device)
        if isinstance(batch, torch.Tensor):
            batch.record_stream(stream)
        elif isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    v.record_stream(stream)

    def _preload(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = self._to_device(batch)

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        self._record_streams(batch)
        self._preload()
        return batch

    def __len__(self):
        return len(self.loader)

    def reset(self):
        """Call at the start of each epoch after sampler.set_epoch()"""
        self.iter = iter(self.loader)
        self._preload()