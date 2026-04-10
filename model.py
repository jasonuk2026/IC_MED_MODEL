"""
model.py

DiseaseAwareEHREncoder — Qwen3-style disease-conditioned patient encoder.

Architecture per sample:
  simple:
    event_embs  (B, N, 768)  ──► input_norm ──► bert_proj_1(768→768) ──► GELU ──► bert_proj_2(768→D) ──► ev_proj (B, N, D)

  transformer:
    event_embs  (B, N, 768)  ──► shallow Transformer ──► ev_hidden (B, N, 768)

  Per-task prefix string: "Please predict disease <name>"
  Tokenised with the Qwen tokenizer, left-padded to a uniform length across all
  tasks so that torch.compile sees static shapes.  The padded token ids are
  embedded once at init via embed_tokens and stored as a (num_tasks, max_P, D)
  buffer — no BioLinkBERT disease projection needed.

  Qwen input sequence:
    [PAD…PAD | prefix+disease tokens | middle tokens | projected events | EOS]

  where middle = " based on the following events.\nStart of medical events:"

  attention_mask = 0 for pad positions, 1 everywhere else.

  Pool: last token (EOS) → L2 normalise

Trainable modules beyond Qwen LoRA:
    input_norm   RMSNorm(bert_dim)               (stabilises event MLP input)
    bert_proj_1  Linear(bert_dim → bert_dim)     (hidden)
    bert_proj_2  Linear(bert_dim → qwen_dim)     (up-project)
"""

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import PeftModel

try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
    _HAS_FLASH_ATTN = True
except Exception:
    flash_attn_varlen_func = None
    pad_input = None
    unpad_input = None
    _HAS_FLASH_ATTN = False

logger = logging.getLogger(__name__)

EOS_TOKEN_ID = 151643  # Qwen3 <|endoftext|>


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RotaryEmbedding requires an even dim, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(device=device))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :].to(dtype=dtype)
        sin = emb.sin()[None, None, :, :].to(dtype=dtype)
        return cos, sin


class QwenStyleMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False).to(dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False).to(dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False).to(dtype)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.up_proj.weight)
        # Zero-init the residual branch output so each block starts near identity.
        nn.init.zeros_(self.down_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ShallowSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dtype: torch.dtype):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be even for rotary embeddings")
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False).to(dtype)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False).to(dtype)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False).to(dtype)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False).to(dtype)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        # Zero-init the residual branch output so attention starts as a no-op.
        nn.init.zeros_(self.o_proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings
        cos = cos.squeeze(0).squeeze(0)[None, :, None, :]
        sin = sin.squeeze(0).squeeze(0)[None, :, None, :]

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if _HAS_FLASH_ATTN:
            attention_mask_bool = attention_mask.bool()
            q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q, attention_mask_bool)
            k_unpad, _, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k, attention_mask_bool)
            v_unpad, _, _, _, _ = unpad_input(v, attention_mask_bool)
            attn_out_unpad = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
            )
            attn_out = pad_input(attn_out_unpad, indices_q, bsz, seq_len)
        else:
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            attn_scores = torch.matmul(q_t, k_t.transpose(-1, -2)) / math.sqrt(self.head_dim)
            key_mask = ~attention_mask.bool()[:, None, None, :]
            attn_scores = attn_scores.masked_fill(key_mask, torch.finfo(attn_scores.dtype).min)
            attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(dtype=hidden_states.dtype)
            attn_out = torch.matmul(attn_probs, v_t).transpose(1, 2)

        attn_out = attn_out * attention_mask[:, :, None, None].to(dtype=hidden_states.dtype)
        attn_out = attn_out.contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(attn_out)


class ShallowTransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, dtype: torch.dtype):
        super().__init__()
        self.self_attn = ShallowSelfAttention(hidden_size, num_heads, dtype)
        self.mlp = QwenStyleMLP(hidden_size, intermediate_size, dtype)
        self.input_layernorm = nn.RMSNorm(hidden_size).to(dtype)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size).to(dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states


class ShallowMLPLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype):
        super().__init__()
        self.mlp = QwenStyleMLP(hidden_size, intermediate_size, dtype)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size).to(dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size).to(dtype)
        self.mlp = QwenStyleMLP(hidden_size, intermediate_size, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class DiseaseAwareEHREncoder(nn.Module):
    """Qwen model conditioned on disease via per-task prefix token embeddings.

    Each task has a fixed prompt prefix "Please predict disease <name>" that is
    tokenised and embedded once at construction time.  All prefixes are
    left-padded to the same length (max across tasks) so that the sequence
    length is static — required for torch.compile with static shapes.

    forward() accepts a task_idx tensor (B,) of integer task indices and looks
    up the corresponding prefix embeddings and attention masks from buffers.

    Sequence layout:
        [PAD…PAD | "Please predict disease <name>" | middle | events | EOS]

    Trainable modules (beyond Qwen LoRA):
        input_norm   RMSNorm(bert_dim)
        bert_proj_1  Linear(bert_dim → bert_dim, no bias)
        bert_proj_2  Linear(bert_dim → qwen_dim, no bias)
    """

    def __init__(
        self,
        qwen_model:       nn.Module,
        bert_dim:         int,
        qwen_dim:         int,
        task_prefix_ids:  torch.Tensor,   # (num_tasks, max_P)  left-padded token ids
        task_prefix_mask: torch.Tensor,   # (num_tasks, max_P)  0=pad, 1=real
        middle_ids:       torch.Tensor,   # (1, P2)
        task_text_embs:   torch.Tensor,   # (num_tasks, bert_dim) BioLinkBERT disease text embeddings
        disease_encoder_type: str = "query_head",
        shallow_encoder_type: str = "transformer",
        shallow_num_layers: int = 2,
        shallow_num_heads: int = 4,
        shallow_intermediate_size: int | None = None,
        disease_head_layers: int = 0,
        disease_head_intermediate_size: int | None = None,
        dtype:            torch.dtype = torch.float32,
    ):
        super().__init__()
        self.qwen        = qwen_model
        self.dtype       = dtype
        self.disease_encoder_type = disease_encoder_type
        self.shallow_encoder_type = shallow_encoder_type
        self.shallow_num_layers = shallow_num_layers
        self.input_norm  = nn.RMSNorm(bert_dim).to(dtype)
        self.bert_proj_1 = nn.Linear(bert_dim, bert_dim, bias=False).to(dtype)
        self.bert_proj_2 = nn.Linear(bert_dim, qwen_dim, bias=False).to(dtype)
        shallow_intermediate_size = shallow_intermediate_size or (bert_dim * 4)
        self.shallow_rotary = RotaryEmbedding(bert_dim // shallow_num_heads)
        if shallow_encoder_type == "transformer":
            self.shallow_layers = nn.ModuleList([
                ShallowTransformerLayer(
                    hidden_size=bert_dim,
                    intermediate_size=shallow_intermediate_size,
                    num_heads=shallow_num_heads,
                    dtype=dtype,
                )
                for _ in range(shallow_num_layers)
            ])
        elif shallow_encoder_type == "mlp":
            self.shallow_layers = nn.ModuleList([
                ShallowMLPLayer(
                    hidden_size=bert_dim,
                    intermediate_size=shallow_intermediate_size,
                    dtype=dtype,
                )
                for _ in range(shallow_num_layers)
            ])
        else:
            self.shallow_layers = nn.ModuleList()
        shallow_out_dim = bert_dim if shallow_encoder_type in {"transformer", "mlp"} else qwen_dim
        disease_head_intermediate_size = disease_head_intermediate_size or (shallow_out_dim * 4)
        num_tasks        = task_prefix_ids.size(0)
        self.task_input_emb = nn.Embedding(num_tasks, bert_dim).to(dtype)
        self.task_input_scale = nn.Parameter(torch.tensor(1e-3, dtype=dtype))
        self.task_cond_emb = nn.Embedding(num_tasks, shallow_out_dim).to(dtype)
        self.cond_fuse_1   = nn.Linear(shallow_out_dim * 2, shallow_out_dim, bias=False).to(dtype)
        self.cond_fuse_2   = nn.Linear(shallow_out_dim, shallow_out_dim, bias=False).to(dtype)
        self.task_res_scale = nn.Parameter(torch.tensor(1e-3, dtype=dtype))
        self.film_gamma = nn.Embedding(num_tasks, shallow_out_dim).to(dtype)
        self.film_beta  = nn.Embedding(num_tasks, shallow_out_dim).to(dtype)
        self.task_query_proj = nn.Linear(bert_dim, shallow_out_dim, bias=False).to(dtype)
        self.disease_head_layers = nn.ModuleList([
            ResidualMLPBlock(
                hidden_size=shallow_out_dim,
                intermediate_size=disease_head_intermediate_size,
                dtype=dtype,
            )
            for _ in range(disease_head_layers)
        ])
        self.task_cross_attn = nn.MultiheadAttention(
            embed_dim=shallow_out_dim,
            num_heads=4,
            batch_first=True,
        ).to(dtype)
        self.task_xattn_scale = nn.Parameter(torch.tensor(1e-3, dtype=dtype))
        self.task_proto_emb = nn.Embedding(num_tasks, shallow_out_dim).to(dtype)
        self.query_fuse = nn.Linear(shallow_out_dim * 2, shallow_out_dim, bias=False).to(dtype)

        nn.init.xavier_uniform_(self.bert_proj_1.weight)
        nn.init.xavier_uniform_(self.bert_proj_2.weight)
        nn.init.normal_(self.task_input_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.task_cond_emb.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.cond_fuse_1.weight)
        nn.init.xavier_uniform_(self.cond_fuse_2.weight)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.xavier_uniform_(self.task_query_proj.weight)
        nn.init.normal_(self.task_proto_emb.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.query_fuse.weight)

        embed_fn = qwen_model.get_input_embeddings()
        with torch.no_grad():
            # Per-task prefix embeddings — frozen, derived from Qwen embed_tokens.
            # Shape: (num_tasks, max_P, D)
            self.register_buffer(
                "task_prefix_embeds",
                embed_fn(task_prefix_ids).to(dtype),
            )
            # Attention mask for prefix positions (0 = pad, 1 = real token).
            # Shape: (num_tasks, max_P)
            self.register_buffer("task_prefix_mask", task_prefix_mask.long())
            self.register_buffer("task_text_embs", task_text_embs.to(dtype))

            self.register_buffer("middle_embeds", embed_fn(middle_ids).to(dtype))  # (1, P2, D)
            eos_ids = torch.full((1, 1), EOS_TOKEN_ID, dtype=torch.long)
            self.register_buffer("eos_embed",    embed_fn(eos_ids).to(dtype))       # (1,  1, D)

    def encode_task_query(self, task_idx: torch.Tensor) -> torch.Tensor:
        if self.disease_encoder_type == "shared_backbone":
            text_emb = self.task_text_embs[task_idx].to(self.dtype)  # (B, bert_dim)
            tokens = text_emb.unsqueeze(1)  # (B, 1, bert_dim)
            mask = torch.ones(tokens.size(0), 1, device=tokens.device, dtype=torch.long)
            if self.shallow_encoder_type == "transformer":
                hidden = tokens
                if self.shallow_num_layers > 0:
                    pos_emb = self.shallow_rotary(hidden.size(1), hidden.device, hidden.dtype)
                    for layer in self.shallow_layers:
                        hidden = layer(hidden, mask, pos_emb)
                return hidden.squeeze(1)
            if self.shallow_encoder_type == "mlp":
                hidden = tokens
                if self.shallow_num_layers > 0:
                    for layer in self.shallow_layers:
                        hidden = layer(hidden, mask)
                return hidden.squeeze(1)
            if self.shallow_encoder_type == "simple":
                normed = self.input_norm(tokens)
                proj = self.bert_proj_2(F.gelu(self.bert_proj_1(normed)))
                return proj.squeeze(1)
            raise ValueError(f"Unknown shallow_encoder_type: {self.shallow_encoder_type}")

        query = self.task_query_proj(self.task_text_embs[task_idx])
        for layer in self.disease_head_layers:
            query = layer(query)
        return query

    def forward(
        self,
        event_embs:   torch.Tensor,         # (B, N, bert_dim)
        event_mask:   torch.Tensor,         # (B, N)            long
        task_idx:     torch.Tensor,         # (B,)              long  — integer task indices
        bypass_qwen:  bool = False,         # Stage 1: skip Qwen, pool ev_proj directly
        condition_on_task: bool = False,    # Lightweight disease-conditioned proj head
        condition_mode: str = "concat",     # concat | residual | film | xattn | xattn_pool | query_proto
        return_pre_emb: bool = False,       # Also return pooled embedding before L2 norm
    ):
        """
        Returns:
            return_pre_emb=False → emb  (B, D)  L2-normalised
            return_pre_emb=True  → (emb, pre_emb) where pre_emb (B, D) is the
                                   mean-pooled representation before L2
                                   normalisation.
        """
        B      = event_embs.size(0)
        device = event_embs.device

        event_embs = event_embs.to(self.dtype)

        if bypass_qwen and condition_on_task and condition_mode == "token_preproj":
            task_input = self.task_input_emb(task_idx).to(event_embs.dtype).unsqueeze(1)
            event_embs = event_embs + self.task_input_scale * task_input

        # 1. Shallow event encoder.
        if self.shallow_encoder_type == "transformer":
            ev_hidden = event_embs
            if self.shallow_num_layers > 0:
                pos_emb = self.shallow_rotary(ev_hidden.size(1), ev_hidden.device, ev_hidden.dtype)
                for layer in self.shallow_layers:
                    ev_hidden = layer(ev_hidden, event_mask, pos_emb)
            shallow_tokens = ev_hidden
        elif self.shallow_encoder_type == "mlp":
            ev_hidden = event_embs
            if self.shallow_num_layers > 0:
                for layer in self.shallow_layers:
                    ev_hidden = layer(ev_hidden, event_mask)
            shallow_tokens = ev_hidden
        elif self.shallow_encoder_type == "simple":
            ev_normed = self.input_norm(event_embs)
            ev_proj = self.bert_proj_2(F.gelu(self.bert_proj_1(ev_normed)))
            shallow_tokens = ev_proj
        else:
            raise ValueError(f"Unknown shallow_encoder_type: {self.shallow_encoder_type}")

        mask_f = event_mask.float().unsqueeze(-1)   # (B, N, 1)

        # ── Stage-1 bypass / shallow modes ────────────────────────────────────
        if bypass_qwen:
            pre_emb = (shallow_tokens * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)  # (B, D)
            if condition_on_task:
                pre_emb = pre_emb.to(self.dtype)
                if condition_mode == "token_preproj":
                    pass
                elif condition_mode == "concat":
                    task_cond = self.task_cond_emb(task_idx).to(pre_emb.dtype)
                    fused = torch.cat([pre_emb, task_cond], dim=-1)
                    pre_emb = self.cond_fuse_2(F.gelu(self.cond_fuse_1(fused)))
                elif condition_mode == "residual":
                    task_cond = self.task_cond_emb(task_idx).to(pre_emb.dtype)
                    pre_emb = pre_emb + self.task_res_scale * task_cond
                elif condition_mode == "film":
                    gamma = self.film_gamma(task_idx).to(pre_emb.dtype)
                    beta  = self.film_beta(task_idx).to(pre_emb.dtype)
                    pre_emb = (1.0 + gamma) * pre_emb + beta
                elif condition_mode == "xattn":
                    task_query = self.encode_task_query(task_idx).to(pre_emb.dtype)
                    attn_out, _ = self.task_cross_attn(
                        query=task_query.unsqueeze(1),
                        key=shallow_tokens,
                        value=shallow_tokens,
                        key_padding_mask=~event_mask.bool(),
                        need_weights=False,
                    )
                    pre_emb = pre_emb + self.task_xattn_scale * attn_out.squeeze(1)
                elif condition_mode == "xattn_pool":
                    task_query = self.encode_task_query(task_idx).to(pre_emb.dtype)
                    attn_out, _ = self.task_cross_attn(
                        query=task_query.unsqueeze(1),
                        key=shallow_tokens,
                        value=shallow_tokens,
                        key_padding_mask=~event_mask.bool(),
                        need_weights=False,
                    )
                    pre_emb = attn_out.squeeze(1)
                elif condition_mode == "query_proto":
                    text_query = self.encode_task_query(task_idx).to(pre_emb.dtype)
                    proto_query = self.task_proto_emb(task_idx).to(pre_emb.dtype)
                    fused_query = self.query_fuse(torch.cat([text_query, proto_query], dim=-1))
                    attn_out, _ = self.task_cross_attn(
                        query=fused_query.unsqueeze(1),
                        key=shallow_tokens,
                        value=shallow_tokens,
                        key_padding_mask=~event_mask.bool(),
                        need_weights=False,
                    )
                    pre_emb = attn_out.squeeze(1)
                else:
                    raise ValueError(f"Unknown condition_mode: {condition_mode}")
            emb     = F.normalize(pre_emb.float(), p=2, dim=-1)
            if return_pre_emb:
                return emb, pre_emb.float()
            return emb

        # 2. Only the Qwen path needs qwen_dim tokens.
        if self.shallow_encoder_type in {"transformer", "mlp"}:
            ev_normed = self.input_norm(shallow_tokens)
            ev_proj   = self.bert_proj_2(F.gelu(self.bert_proj_1(ev_normed))) # (B, N, qwen_dim)
        else:
            ev_proj = shallow_tokens

        # 3. Per-task prefix embeddings and masks (looked up from buffers)
        prefix      = self.task_prefix_embeds[task_idx]   # (B, max_P, D)
        prefix_mask = self.task_prefix_mask[task_idx]     # (B, max_P)
        middle      = self.middle_embeds.expand(B, -1, -1)
        eos         = self.eos_embed.expand(B, -1, -1)

        # 3. Assemble: [prefix+disease | middle | events | EOS]
        inputs_embeds = torch.cat([prefix, middle, ev_proj, eos], dim=1)

        P2 = middle.size(1)
        ones = lambda n: torch.ones(B, n, dtype=torch.long, device=device)
        attention_mask = torch.cat([prefix_mask, ones(P2), event_mask, ones(1)], dim=1)

        # 4. Forward through Qwen (inputs_embeds bypasses embed_tokens)
        out = self.qwen(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        # 5. Mean pool over event hidden states → L2 normalise.
        # Event tokens occupy positions [max_P + P2 : max_P + P2 + N] in the sequence.
        hidden   = out.last_hidden_state                       # (B, T, D)
        max_P    = prefix.size(1)
        P_len    = max_P + P2                                  # sequence offset of first event
        N        = event_embs.size(1)
        ev_hid   = hidden[:, P_len : P_len + N, :]            # (B, N, D)
        pre_emb  = (ev_hid * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)  # (B, D)
        emb = F.normalize(pre_emb.float(), p=2, dim=-1)
        if return_pre_emb:
            return emb, pre_emb.float()
        return emb

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def save_checkpoint(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        raw_qwen = self.qwen.module if isinstance(self.qwen, DDP) else self.qwen
        raw_qwen.save_pretrained(str(save_dir / "lora"))
        torch.save(
            {
                "bert_proj_1": self.bert_proj_1.state_dict(),
                "bert_proj_2": self.bert_proj_2.state_dict(),
                "input_norm":  self.input_norm.state_dict(),
                "shallow_layers": self.shallow_layers.state_dict(),
                "shallow_encoder_type": self.shallow_encoder_type,
                "disease_encoder_type": self.disease_encoder_type,
                "task_input_emb": self.task_input_emb.state_dict(),
                "task_input_scale": self.task_input_scale.detach().cpu(),
                "task_cond_emb": self.task_cond_emb.state_dict(),
                "cond_fuse_1":   self.cond_fuse_1.state_dict(),
                "cond_fuse_2":   self.cond_fuse_2.state_dict(),
                "task_res_scale": self.task_res_scale.detach().cpu(),
                "film_gamma": self.film_gamma.state_dict(),
                "film_beta":  self.film_beta.state_dict(),
                "task_query_proj": self.task_query_proj.state_dict(),
                "disease_head_layers": self.disease_head_layers.state_dict(),
                "task_cross_attn": self.task_cross_attn.state_dict(),
                "task_xattn_scale": self.task_xattn_scale.detach().cpu(),
                "task_proto_emb": self.task_proto_emb.state_dict(),
                "query_fuse": self.query_fuse.state_dict(),
            },
            save_dir / "extra_modules.pt",
        )
        logger.info(f"  Saved checkpoint → {save_dir}")

    @classmethod
    def load_from_checkpoint(
        cls,
        save_dir:         Path,
        qwen_base:        nn.Module,
        bert_dim:         int,
        qwen_dim:         int,
        task_prefix_ids:  torch.Tensor,
        task_prefix_mask: torch.Tensor,
        middle_ids:       torch.Tensor,
        task_text_embs:   torch.Tensor,
        disease_encoder_type: str = "query_head",
        shallow_num_layers: int = 2,
        shallow_num_heads: int = 4,
        shallow_intermediate_size: int | None = None,
        disease_head_layers: int = 0,
        disease_head_intermediate_size: int | None = None,
        dtype:            torch.dtype = torch.float32,
    ) -> "DiseaseAwareEHREncoder":
        qwen_lora = PeftModel.from_pretrained(qwen_base, str(save_dir / "lora"))
        encoder   = cls(
            qwen_lora, bert_dim, qwen_dim,
            task_prefix_ids, task_prefix_mask, middle_ids, task_text_embs,
            disease_encoder_type=disease_encoder_type,
            disease_head_layers=disease_head_layers,
            disease_head_intermediate_size=disease_head_intermediate_size,
            shallow_num_layers=shallow_num_layers,
            shallow_num_heads=shallow_num_heads,
            shallow_intermediate_size=shallow_intermediate_size,
            dtype=dtype,
        )
        extra = torch.load(save_dir / "extra_modules.pt", map_location="cpu")
        encoder.bert_proj_1.load_state_dict(extra["bert_proj_1"])
        encoder.bert_proj_2.load_state_dict(extra["bert_proj_2"])
        if "input_norm" in extra:
            encoder.input_norm.load_state_dict(extra["input_norm"])
        if "shallow_layers" in extra:
            encoder.shallow_layers.load_state_dict(extra["shallow_layers"])
        if "task_input_emb" in extra:
            encoder.task_input_emb.load_state_dict(extra["task_input_emb"])
        if "task_input_scale" in extra:
            encoder.task_input_scale.data.copy_(extra["task_input_scale"].to(dtype))
        if "task_cond_emb" in extra:
            encoder.task_cond_emb.load_state_dict(extra["task_cond_emb"])
        if "cond_fuse_1" in extra:
            encoder.cond_fuse_1.load_state_dict(extra["cond_fuse_1"])
        if "cond_fuse_2" in extra:
            encoder.cond_fuse_2.load_state_dict(extra["cond_fuse_2"])
        if "task_res_scale" in extra:
            encoder.task_res_scale.data.copy_(extra["task_res_scale"].to(dtype))
        if "film_gamma" in extra:
            encoder.film_gamma.load_state_dict(extra["film_gamma"])
        if "film_beta" in extra:
            encoder.film_beta.load_state_dict(extra["film_beta"])
        if "task_query_proj" in extra:
            encoder.task_query_proj.load_state_dict(extra["task_query_proj"])
        if "disease_head_layers" in extra:
            encoder.disease_head_layers.load_state_dict(extra["disease_head_layers"])
        if "task_cross_attn" in extra:
            encoder.task_cross_attn.load_state_dict(extra["task_cross_attn"])
        if "task_xattn_scale" in extra:
            encoder.task_xattn_scale.data.copy_(extra["task_xattn_scale"].to(dtype))
        if "task_proto_emb" in extra:
            encoder.task_proto_emb.load_state_dict(extra["task_proto_emb"])
        if "query_fuse" in extra:
            encoder.query_fuse.load_state_dict(extra["query_fuse"])
        return encoder
