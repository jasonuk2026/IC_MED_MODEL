from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size).to(dtype)
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False).to(dtype)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False).to(dtype)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class DiseaseEventConcatClassifier(nn.Module):
    def __init__(
        self,
        *,
        bert_dim: int,
        task_text_embs: torch.Tensor,
        hidden_size: int = 768,
        patient_layers: int = 1,
        head_layers: int = 1,
        intermediate_size: int | None = None,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.bert_dim = bert_dim
        self.hidden_size = hidden_size
        self.patient_layers_count = patient_layers
        self.head_layers_count = head_layers
        self.intermediate_size = intermediate_size or (hidden_size * 4)
        self.dropout_p = dropout
        self.dtype = dtype

        self.register_buffer("task_text_embs", task_text_embs.to(dtype))

        self.patient_norm = nn.RMSNorm(bert_dim).to(dtype)
        self.patient_proj = nn.Linear(bert_dim, hidden_size, bias=False).to(dtype)
        nn.init.xavier_uniform_(self.patient_proj.weight)
        self.patient_blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_size, self.intermediate_size, dtype)
            for _ in range(patient_layers)
        ])

        self.disease_norm = nn.RMSNorm(bert_dim).to(dtype)
        self.disease_proj = nn.Linear(bert_dim, hidden_size, bias=False).to(dtype)
        nn.init.xavier_uniform_(self.disease_proj.weight)

        self.fuse_norm = nn.RMSNorm(hidden_size * 2).to(dtype)
        self.fuse_in = nn.Linear(hidden_size * 2, hidden_size, bias=False).to(dtype)
        nn.init.xavier_uniform_(self.fuse_in.weight)
        self.head_blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_size, self.intermediate_size, dtype)
            for _ in range(head_layers)
        ])
        self.out_norm = nn.RMSNorm(hidden_size).to(dtype)
        self.out_proj = nn.Linear(hidden_size, 1, bias=False).to(dtype)
        nn.init.zeros_(self.out_proj.weight)
        self.dropout = nn.Dropout(dropout)

    def encode_patient(self, event_embs: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
        x = self.patient_proj(self.patient_norm(event_embs.to(self.dtype)))
        for block in self.patient_blocks:
            x = block(x)
        mask_f = event_mask.to(x.dtype).unsqueeze(-1)
        pooled = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        return pooled

    def encode_disease(self, task_idx: torch.Tensor) -> torch.Tensor:
        text_emb = self.task_text_embs[task_idx].to(self.dtype)
        return self.disease_proj(self.disease_norm(text_emb))

    def forward(
        self,
        event_embs: torch.Tensor,
        event_mask: torch.Tensor,
        task_idx: torch.Tensor,
        return_features: bool = False,
    ):
        patient = self.encode_patient(event_embs, event_mask)
        disease = self.encode_disease(task_idx)
        fused = torch.cat([patient, disease], dim=-1)
        hidden = self.fuse_in(self.fuse_norm(fused))
        hidden = self.dropout(hidden)
        for block in self.head_blocks:
            hidden = block(hidden)
        logits = self.out_proj(self.out_norm(hidden)).squeeze(-1).float()
        if return_features:
            return logits, patient.float(), disease.float(), hidden.float()
        return logits

    def save_checkpoint(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "bert_dim": self.bert_dim,
                    "hidden_size": self.hidden_size,
                    "patient_layers": self.patient_layers_count,
                    "head_layers": self.head_layers_count,
                    "intermediate_size": self.intermediate_size,
                    "dropout": self.dropout_p,
                    "dtype": str(self.dtype),
                },
            },
            save_dir / "model.pt",
        )
        logger.info("  Saved checkpoint -> %s", save_dir)

    @classmethod
    def load_checkpoint(
        cls,
        save_dir: Path,
        *,
        task_text_embs: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "DiseaseEventConcatClassifier":
        payload = torch.load(save_dir / "model.pt", map_location="cpu")
        cfg = payload["config"]
        model = cls(
            bert_dim=cfg["bert_dim"],
            task_text_embs=task_text_embs,
            hidden_size=cfg["hidden_size"],
            patient_layers=cfg["patient_layers"],
            head_layers=cfg["head_layers"],
            intermediate_size=cfg["intermediate_size"],
            dropout=cfg["dropout"],
            dtype=dtype,
        )
        model.load_state_dict(payload["state_dict"])
        return model.to(device)
