"""
model.py

DiseaseAwareEHREncoder — Qwen3-style disease-conditioned patient encoder.

Architecture per sample:
  event_embs  (B, N, 768)  ──► input_norm ──► bert_proj_1(768→768) ──► GELU ──► bert_proj_2(768→D) ──► ev_proj (B, N, D)

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
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import PeftModel

logger = logging.getLogger(__name__)

EOS_TOKEN_ID = 151643  # Qwen3 <|endoftext|>


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
        dtype:            torch.dtype = torch.float32,
    ):
        super().__init__()
        self.qwen        = qwen_model
        self.dtype       = dtype
        self.input_norm  = nn.RMSNorm(bert_dim).to(dtype)
        self.bert_proj_1 = nn.Linear(bert_dim, bert_dim, bias=False).to(dtype)
        self.bert_proj_2 = nn.Linear(bert_dim, qwen_dim, bias=False).to(dtype)
        num_tasks        = task_prefix_ids.size(0)
        self.task_input_emb = nn.Embedding(num_tasks, bert_dim).to(dtype)
        self.task_input_scale = nn.Parameter(torch.tensor(1e-3, dtype=dtype))
        self.task_cond_emb = nn.Embedding(num_tasks, qwen_dim).to(dtype)
        self.cond_fuse_1   = nn.Linear(qwen_dim * 2, qwen_dim, bias=False).to(dtype)
        self.cond_fuse_2   = nn.Linear(qwen_dim, qwen_dim, bias=False).to(dtype)
        self.task_res_scale = nn.Parameter(torch.tensor(1e-3, dtype=dtype))
        self.film_gamma = nn.Embedding(num_tasks, qwen_dim).to(dtype)
        self.film_beta  = nn.Embedding(num_tasks, qwen_dim).to(dtype)
        self.task_query_proj = nn.Linear(bert_dim, qwen_dim, bias=False).to(dtype)
        self.task_cross_attn = nn.MultiheadAttention(
            embed_dim=qwen_dim,
            num_heads=4,
            batch_first=True,
        ).to(dtype)
        self.task_xattn_scale = nn.Parameter(torch.tensor(1e-3, dtype=dtype))
        self.task_proto_emb = nn.Embedding(num_tasks, qwen_dim).to(dtype)
        self.query_fuse = nn.Linear(qwen_dim * 2, qwen_dim, bias=False).to(dtype)

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

        # 1. RMSNorm → MLP projection for events (Qwen3-style: norm before linear)
        ev_normed = self.input_norm(event_embs)                            # (B, N, bert_dim)
        ev_proj   = self.bert_proj_2(F.gelu(self.bert_proj_1(ev_normed))) # (B, N, qwen_dim)

        mask_f = event_mask.float().unsqueeze(-1)   # (B, N, 1)

        # ── Stage-1 bypass / shallow modes ────────────────────────────────────
        if bypass_qwen:
            pre_emb = (ev_proj * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)  # (B, D)
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
                    task_query = self.task_query_proj(self.task_text_embs[task_idx]).to(pre_emb.dtype)
                    attn_out, _ = self.task_cross_attn(
                        query=task_query.unsqueeze(1),
                        key=ev_proj,
                        value=ev_proj,
                        key_padding_mask=~event_mask.bool(),
                        need_weights=False,
                    )
                    pre_emb = pre_emb + self.task_xattn_scale * attn_out.squeeze(1)
                elif condition_mode == "xattn_pool":
                    task_query = self.task_query_proj(self.task_text_embs[task_idx]).to(pre_emb.dtype)
                    attn_out, _ = self.task_cross_attn(
                        query=task_query.unsqueeze(1),
                        key=ev_proj,
                        value=ev_proj,
                        key_padding_mask=~event_mask.bool(),
                        need_weights=False,
                    )
                    pre_emb = attn_out.squeeze(1)
                elif condition_mode == "query_proto":
                    text_query = self.task_query_proj(self.task_text_embs[task_idx]).to(pre_emb.dtype)
                    proto_query = self.task_proto_emb(task_idx).to(pre_emb.dtype)
                    fused_query = self.query_fuse(torch.cat([text_query, proto_query], dim=-1))
                    attn_out, _ = self.task_cross_attn(
                        query=fused_query.unsqueeze(1),
                        key=ev_proj,
                        value=ev_proj,
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

        # 2. Per-task prefix embeddings and masks (looked up from buffers)
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
                "task_input_emb": self.task_input_emb.state_dict(),
                "task_input_scale": self.task_input_scale.detach().cpu(),
                "task_cond_emb": self.task_cond_emb.state_dict(),
                "cond_fuse_1":   self.cond_fuse_1.state_dict(),
                "cond_fuse_2":   self.cond_fuse_2.state_dict(),
                "task_res_scale": self.task_res_scale.detach().cpu(),
                "film_gamma": self.film_gamma.state_dict(),
                "film_beta":  self.film_beta.state_dict(),
                "task_query_proj": self.task_query_proj.state_dict(),
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
        dtype:            torch.dtype = torch.float32,
    ) -> "DiseaseAwareEHREncoder":
        qwen_lora = PeftModel.from_pretrained(qwen_base, str(save_dir / "lora"))
        encoder   = cls(
            qwen_lora, bert_dim, qwen_dim,
            task_prefix_ids, task_prefix_mask, middle_ids, task_text_embs,
            dtype=dtype,
        )
        extra = torch.load(save_dir / "extra_modules.pt", map_location="cpu")
        encoder.bert_proj_1.load_state_dict(extra["bert_proj_1"])
        encoder.bert_proj_2.load_state_dict(extra["bert_proj_2"])
        if "input_norm" in extra:
            encoder.input_norm.load_state_dict(extra["input_norm"])
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
        if "task_cross_attn" in extra:
            encoder.task_cross_attn.load_state_dict(extra["task_cross_attn"])
        if "task_xattn_scale" in extra:
            encoder.task_xattn_scale.data.copy_(extra["task_xattn_scale"].to(dtype))
        if "task_proto_emb" in extra:
            encoder.task_proto_emb.load_state_dict(extra["task_proto_emb"])
        if "query_fuse" in extra:
            encoder.query_fuse.load_state_dict(extra["query_fuse"])
        return encoder
