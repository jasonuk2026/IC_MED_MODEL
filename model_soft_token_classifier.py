from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_positions: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"Rotary head_dim must be even, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_positions).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(2)
        return cos, sin


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout: float,
        dtype: torch.dtype,
        position_type: str,
        max_positions: int,
        causal: bool,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.attn_norm = nn.RMSNorm(hidden_size).to(dtype)
        self.position_type = position_type
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal
        if position_type == "rotary":
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False).to(dtype)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False).to(dtype)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False).to(dtype)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False).to(dtype)
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.zeros_(self.o_proj.weight)
            self.rotary = RotaryEmbedding(self.head_dim, max_positions=max_positions)
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                batch_first=True,
                dropout=dropout,
                bias=False,
            ).to(dtype)
        self.mlp_norm = nn.RMSNorm(hidden_size).to(dtype)
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False).to(dtype)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False).to(dtype)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc2.weight)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        return_attn_probs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        residual = x
        attn_in = self.attn_norm(x)
        seq_len = attn_in.size(1)
        causal_mask = None
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=attn_in.device, dtype=torch.bool),
                diagonal=1,
            )
        if self.position_type == "rotary":
            bsz, seq_len, hidden = attn_in.shape
            q = self.q_proj(attn_in).view(bsz, seq_len, self.num_heads, self.head_dim)
            k = self.k_proj(attn_in).view(bsz, seq_len, self.num_heads, self.head_dim)
            v = self.v_proj(attn_in).view(bsz, seq_len, self.num_heads, self.head_dim)
            cos, sin = self.rotary.get_cos_sin(seq_len, attn_in.device, attn_in.dtype)
            q = _apply_rotary(q, cos, sin)
            k = _apply_rotary(k, cos, sin)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if key_padding_mask is not None:
                scores = scores.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
            if causal_mask is not None:
                scores = scores.masked_fill(causal_mask[None, None, :, :], torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            attn_out = torch.matmul(probs, v).permute(0, 2, 1, 3).reshape(bsz, seq_len, hidden)
            attn_out = self.o_proj(attn_out)
            attn_probs = probs
        else:
            attn_out, attn_probs = self.attn(
                attn_in,
                attn_in,
                attn_in,
                attn_mask=causal_mask,
                key_padding_mask=key_padding_mask,
                need_weights=return_attn_probs,
                average_attn_weights=False,
            )
        x = residual + self.dropout(attn_out)

        residual = x
        mlp_in = self.mlp_norm(x)
        mlp_out = self.fc2(F.gelu(self.fc1(mlp_in)))
        x = residual + self.dropout(mlp_out)
        if return_attn_probs:
            return x, attn_probs
        return x


class DiseaseEventSoftTokenClassifier(nn.Module):
    def __init__(
        self,
        *,
        bert_dim: int,
        task_text_embs: torch.Tensor,
        hidden_size: int = 768,
        num_layers: int = 1,
        num_heads: int = 4,
        intermediate_size: int | None = None,
        head_layers: int = 1,
        max_positions: int = 1024,
        position_type: str = "learned",
        attention_type: str = "bidirectional",
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.bert_dim = bert_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size or (hidden_size * 4)
        self.head_layers_count = head_layers
        self.max_positions = max_positions
        self.position_type = position_type
        self.attention_type = attention_type
        self.dropout_p = dropout
        self.dtype = dtype

        self.register_buffer("task_text_embs", task_text_embs.to(dtype))

        self.event_norm = nn.RMSNorm(bert_dim).to(dtype)
        self.event_proj = nn.Linear(bert_dim, hidden_size, bias=False).to(dtype)
        nn.init.xavier_uniform_(self.event_proj.weight)

        self.disease_norm = nn.RMSNorm(bert_dim).to(dtype)
        self.disease_proj = nn.Linear(bert_dim, hidden_size, bias=False).to(dtype)
        nn.init.xavier_uniform_(self.disease_proj.weight)

        if position_type == "learned":
            self.pos_emb = nn.Embedding(max_positions, hidden_size).to(dtype)
            nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        else:
            self.pos_emb = None

        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size,
                self.intermediate_size,
                num_heads,
                dropout,
                dtype,
                position_type,
                max_positions + 1,
                causal=(attention_type == "causal"),
            )
            for _ in range(num_layers)
        ])

        self.head_norm = nn.RMSNorm(hidden_size).to(dtype)
        self.head_in = nn.Linear(hidden_size, hidden_size, bias=False).to(dtype)
        nn.init.xavier_uniform_(self.head_in.weight)
        self.head_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size,
                self.intermediate_size,
                num_heads,
                dropout,
                dtype,
                "learned",
                max_positions + 1,
                causal=False,
            )
            for _ in range(head_layers)
        ])
        self.out_norm = nn.RMSNorm(hidden_size).to(dtype)
        self.out_proj = nn.Linear(hidden_size, 1, bias=False).to(dtype)
        nn.init.zeros_(self.out_proj.weight)

        self.aux_norm = nn.RMSNorm(hidden_size).to(dtype)
        self.aux_in = nn.Linear(hidden_size, hidden_size, bias=False).to(dtype)
        nn.init.xavier_uniform_(self.aux_in.weight)
        self.aux_out_norm = nn.RMSNorm(hidden_size).to(dtype)
        self.aux_out_proj = nn.Linear(hidden_size, 1, bias=False).to(dtype)
        nn.init.zeros_(self.aux_out_proj.weight)
        self.dropout = nn.Dropout(dropout)

    def _left_pad_events(self, event_tokens: torch.Tensor, event_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Input batches are right-padded by the collator; shift valid events to the right
        # so the disease token sits immediately after the real event suffix.
        bsz, seq_len, hidden = event_tokens.shape
        aligned_tokens = torch.zeros_like(event_tokens)
        aligned_mask = torch.zeros_like(event_mask)
        lengths = event_mask.long().sum(dim=1)
        for i in range(bsz):
            length = int(lengths[i].item())
            if length <= 0:
                continue
            aligned_tokens[i, seq_len - length :] = event_tokens[i, :length]
            aligned_mask[i, seq_len - length :] = 1
        return aligned_tokens, aligned_mask

    def forward(
        self,
        event_embs: torch.Tensor,
        event_mask: torch.Tensor,
        task_idx: torch.Tensor,
        return_features: bool = False,
        return_aux_logits: bool = False,
    ):
        event_tokens = self.event_proj(self.event_norm(event_embs.to(self.dtype)))
        event_tokens, event_mask = self._left_pad_events(event_tokens, event_mask)
        disease_token = self.disease_proj(self.disease_norm(self.task_text_embs[task_idx].to(self.dtype))).unsqueeze(1)
        event_seq_len = event_tokens.size(1)
        if event_seq_len > self.max_positions:
            raise ValueError(f"Event sequence length {event_seq_len} exceeds max_positions={self.max_positions}")
        if self.position_type == "learned":
            pos_ids = torch.arange(event_seq_len, device=event_tokens.device)
            event_pos = self.pos_emb(pos_ids).unsqueeze(0)
            event_tokens = event_tokens + event_pos * event_mask.to(event_tokens.dtype).unsqueeze(-1)
        tokens = torch.cat([event_tokens, disease_token], dim=1)

        batch_size = event_mask.size(0)
        disease_mask = torch.ones(batch_size, 1, device=event_mask.device, dtype=event_mask.dtype)
        full_mask = torch.cat([event_mask, disease_mask], dim=1)
        key_padding_mask = ~full_mask.bool()

        hidden = tokens
        for layer in self.layers:
            hidden = layer(hidden, key_padding_mask=key_padding_mask)

        event_hidden = hidden[:, :-1]
        disease_hidden = hidden[:, -1]
        head_hidden = self.head_in(self.head_norm(disease_hidden))
        head_hidden = self.dropout(head_hidden)
        for block in self.head_blocks:
            head_hidden = block(head_hidden.unsqueeze(1), key_padding_mask=None).squeeze(1)

        logits = self.out_proj(self.out_norm(head_hidden)).squeeze(-1).float()
        mask_f = event_mask.to(event_hidden.dtype).unsqueeze(-1)
        event_pooled = (event_hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        aux_hidden = self.dropout(self.aux_in(self.aux_norm(event_pooled)))
        aux_logits = self.aux_out_proj(self.aux_out_norm(aux_hidden)).squeeze(-1).float()
        if return_features:
            return logits, aux_logits, disease_hidden.float(), head_hidden.float(), event_pooled.float()
        if return_aux_logits:
            return logits, aux_logits, disease_hidden.float(), event_pooled.float()
        return logits

    @torch.inference_mode()
    def get_disease_event_attention(
        self,
        event_embs: torch.Tensor,
        event_mask: torch.Tensor,
        task_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        event_tokens = self.event_proj(self.event_norm(event_embs.to(self.dtype)))
        event_tokens, event_mask = self._left_pad_events(event_tokens, event_mask)
        disease_input = self.disease_proj(self.disease_norm(self.task_text_embs[task_idx].to(self.dtype)))
        disease_token = disease_input.unsqueeze(1)
        event_seq_len = event_tokens.size(1)
        if event_seq_len > self.max_positions:
            raise ValueError(f"Event sequence length {event_seq_len} exceeds max_positions={self.max_positions}")
        if self.position_type == "learned":
            pos_ids = torch.arange(event_seq_len, device=event_tokens.device)
            event_pos = self.pos_emb(pos_ids).unsqueeze(0)
            event_tokens = event_tokens + event_pos * event_mask.to(event_tokens.dtype).unsqueeze(-1)
        tokens = torch.cat([event_tokens, disease_token], dim=1)

        batch_size = event_mask.size(0)
        disease_mask = torch.ones(batch_size, 1, device=event_mask.device, dtype=event_mask.dtype)
        full_mask = torch.cat([event_mask, disease_mask], dim=1)
        key_padding_mask = ~full_mask.bool()

        hidden = tokens
        last_attn = None
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                hidden, last_attn = layer(hidden, key_padding_mask=key_padding_mask, return_attn_probs=True)
            else:
                hidden = layer(hidden, key_padding_mask=key_padding_mask)

        disease_hidden = hidden[:, -1].float()
        event_hidden = hidden[:, :-1].float()
        event_pooled = (event_hidden * event_mask.unsqueeze(-1).float()).sum(1) / event_mask.sum(1, keepdim=True).clamp(min=1).float()

        # last_attn: [B, H, T, T], disease token is the last query position.
        disease_attn = last_attn[:, :, -1, :-1].float()
        disease_attn = disease_attn * event_mask.unsqueeze(1).float()
        attn_sum = disease_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        disease_attn = disease_attn / attn_sum
        mean_attn = disease_attn.mean(dim=1)

        return {
            "disease_input": disease_input.float(),
            "disease_hidden": disease_hidden,
            "event_hidden": event_hidden,
            "event_pooled": event_pooled,
            "event_mask": event_mask.float(),
            "attn_per_head": disease_attn,
            "attn_mean": mean_attn,
        }

    def save_checkpoint(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "bert_dim": self.bert_dim,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "num_heads": self.num_heads,
                    "intermediate_size": self.intermediate_size,
                    "head_layers": self.head_layers_count,
                    "max_positions": self.max_positions,
                    "position_type": self.position_type,
                    "attention_type": self.attention_type,
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
    ) -> "DiseaseEventSoftTokenClassifier":
        payload = torch.load(save_dir / "model.pt", map_location="cpu")
        cfg = payload["config"]
        model = cls(
            bert_dim=cfg["bert_dim"],
            task_text_embs=task_text_embs,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            intermediate_size=cfg["intermediate_size"],
            head_layers=cfg["head_layers"],
            max_positions=cfg.get("max_positions", 1024),
            position_type=cfg.get("position_type", "learned"),
            attention_type=cfg.get("attention_type", "bidirectional"),
            dropout=cfg["dropout"],
            dtype=dtype,
        )
        model.load_state_dict(payload["state_dict"])
        return model.to(device)
