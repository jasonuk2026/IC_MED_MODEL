"""
model_jepa.py

JEPA-style disease-to-patient retrieval model.

- Patient events go through an online shared backbone, are pooled, then passed
  through a predictor head.
- Disease text embeddings go through an EMA teacher copy of the same backbone.
- Training aligns patient predictions to teacher disease targets with the same
  listwise retrieval objective used by the baseline retrieval model.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import RotaryEmbedding, ShallowMLPLayer, ShallowTransformerLayer, ResidualMLPBlock

logger = logging.getLogger(__name__)


class SharedEventBackbone(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        encoder_type: str,
        num_layers: int,
        num_heads: int,
        intermediate_size: int | None,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder_type = encoder_type
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dtype = dtype

        self.input_norm = nn.RMSNorm(hidden_size).to(dtype)
        self.proj_1 = nn.Linear(hidden_size, hidden_size, bias=False).to(dtype)
        self.proj_2 = nn.Linear(hidden_size, hidden_size, bias=False).to(dtype)
        nn.init.xavier_uniform_(self.proj_1.weight)
        nn.init.xavier_uniform_(self.proj_2.weight)

        intermediate_size = intermediate_size or (hidden_size * 4)
        self.rotary = RotaryEmbedding(hidden_size // num_heads)
        self.simple_residual_layers = nn.ModuleList()

        if encoder_type == "transformer":
            self.layers = nn.ModuleList([
                ShallowTransformerLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ])
        elif encoder_type == "mlp":
            self.layers = nn.ModuleList([
                ShallowMLPLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ])
        elif encoder_type == "simple":
            self.layers = nn.ModuleList()
            self.simple_residual_layers = nn.ModuleList([
                ResidualMLPBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def encode_tokens(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = tokens.to(self.dtype)
        mask = mask.long()

        if self.encoder_type == "transformer":
            hidden = tokens
            if self.num_layers > 0:
                pos_emb = self.rotary(hidden.size(1), hidden.device, hidden.dtype)
                for layer in self.layers:
                    hidden = layer(hidden, mask, pos_emb)
            return hidden

        if self.encoder_type == "mlp":
            hidden = tokens
            if self.num_layers > 0:
                for layer in self.layers:
                    hidden = layer(hidden, mask)
            return hidden

        hidden = self.proj_2(F.gelu(self.proj_1(self.input_norm(tokens))))
        if self.num_layers > 0:
            for layer in self.simple_residual_layers:
                hidden = layer(hidden)
        return hidden * mask.unsqueeze(-1).to(hidden.dtype)

    def encode_pooled(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encode_tokens(tokens, mask)
        mask_f = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)


class DiseasePatientJEPAModel(nn.Module):
    def __init__(
        self,
        *,
        bert_dim: int,
        task_text_embs: torch.Tensor,
        shallow_encoder_type: str = "simple",
        shallow_num_layers: int = 0,
        shallow_num_heads: int = 4,
        shallow_intermediate_size: int | None = None,
        predictor_layers: int = 1,
        predictor_intermediate_size: int | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.bert_dim = bert_dim
        self.dtype = dtype
        self.shallow_encoder_type = shallow_encoder_type
        self.shallow_num_layers = shallow_num_layers
        self.shallow_num_heads = shallow_num_heads
        self.shallow_intermediate_size = shallow_intermediate_size or (bert_dim * 4)
        self.predictor_layers_count = predictor_layers
        self.predictor_intermediate_size = predictor_intermediate_size or (bert_dim * 4)

        self.online_backbone = SharedEventBackbone(
            hidden_size=bert_dim,
            encoder_type=shallow_encoder_type,
            num_layers=shallow_num_layers,
            num_heads=shallow_num_heads,
            intermediate_size=self.shallow_intermediate_size,
            dtype=dtype,
        )
        self.teacher_backbone = copy.deepcopy(self.online_backbone)
        for p in self.teacher_backbone.parameters():
            p.requires_grad_(False)

        self.predictor = nn.ModuleList([
            ResidualMLPBlock(
                hidden_size=bert_dim,
                intermediate_size=self.predictor_intermediate_size,
                dtype=dtype,
            )
            for _ in range(predictor_layers)
        ])
        self.register_buffer("task_text_embs", task_text_embs.to(dtype))

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        for teacher_param, online_param in zip(self.teacher_backbone.parameters(), self.online_backbone.parameters()):
            teacher_param.data.mul_(momentum).add_(online_param.data, alpha=1.0 - momentum)

    def _apply_predictor(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.predictor:
            x = layer(x)
        return x

    def encode_patient_online(
        self,
        event_embs: torch.Tensor,
        event_mask: torch.Tensor,
        *,
        return_pre: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        pre = self.online_backbone.encode_pooled(event_embs, event_mask)
        pred = self._apply_predictor(pre)
        emb = F.normalize(pred.float(), p=2, dim=-1)
        if return_pre:
            return emb, pred.float()
        return emb

    @torch.no_grad()
    def encode_patient_teacher(
        self,
        event_embs: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        pre = self.teacher_backbone.encode_pooled(event_embs, event_mask)
        return F.normalize(pre.float(), p=2, dim=-1)

    def encode_disease_online(
        self,
        task_idx: torch.Tensor,
        *,
        return_pre: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        text = self.task_text_embs[task_idx].to(self.dtype).unsqueeze(1)
        mask = torch.ones(text.size(0), 1, device=text.device, dtype=torch.long)
        pre = self.online_backbone.encode_pooled(text, mask)
        pred = self._apply_predictor(pre)
        emb = F.normalize(pred.float(), p=2, dim=-1)
        if return_pre:
            return emb, pred.float()
        return emb

    @torch.no_grad()
    def encode_disease_teacher(
        self,
        task_idx: torch.Tensor,
        *,
        return_pre: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        text = self.task_text_embs[task_idx].to(self.dtype).unsqueeze(1)
        mask = torch.ones(text.size(0), 1, device=text.device, dtype=torch.long)
        pre = self.teacher_backbone.encode_pooled(text, mask)
        emb = F.normalize(pre.float(), p=2, dim=-1)
        if return_pre:
            return emb, pre.float()
        return emb

    def save_checkpoint(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "bert_dim": self.bert_dim,
                    "shallow_encoder_type": self.shallow_encoder_type,
                    "shallow_num_layers": self.shallow_num_layers,
                    "shallow_num_heads": self.shallow_num_heads,
                    "shallow_intermediate_size": self.shallow_intermediate_size,
                    "predictor_layers": self.predictor_layers_count,
                    "predictor_intermediate_size": self.predictor_intermediate_size,
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
    ) -> "DiseasePatientJEPAModel":
        payload = torch.load(save_dir / "model.pt", map_location="cpu")
        config = payload["config"]
        model = cls(
            bert_dim=config["bert_dim"],
            task_text_embs=task_text_embs,
            shallow_encoder_type=config["shallow_encoder_type"],
            shallow_num_layers=config["shallow_num_layers"],
            shallow_num_heads=config["shallow_num_heads"],
            shallow_intermediate_size=config["shallow_intermediate_size"],
            predictor_layers=config["predictor_layers"],
            predictor_intermediate_size=config["predictor_intermediate_size"],
            dtype=dtype,
        )
        model.load_state_dict(payload["state_dict"])
        model = model.to(device)
        return model
