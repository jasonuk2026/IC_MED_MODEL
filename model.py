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
        dtype:            torch.dtype = torch.float32,
    ):
        super().__init__()
        self.qwen        = qwen_model
        self.dtype       = dtype
        self.input_norm  = nn.RMSNorm(bert_dim).to(dtype)
        self.bert_proj_1 = nn.Linear(bert_dim, bert_dim, bias=False).to(dtype)
        self.bert_proj_2 = nn.Linear(bert_dim, qwen_dim, bias=False).to(dtype)

        nn.init.xavier_uniform_(self.bert_proj_1.weight)
        nn.init.xavier_uniform_(self.bert_proj_2.weight)

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

            self.register_buffer("middle_embeds", embed_fn(middle_ids).to(dtype))  # (1, P2, D)
            eos_ids = torch.full((1, 1), EOS_TOKEN_ID, dtype=torch.long)
            self.register_buffer("eos_embed",    embed_fn(eos_ids).to(dtype))       # (1,  1, D)

    def forward(
        self,
        event_embs:   torch.Tensor,         # (B, N, bert_dim)
        event_mask:   torch.Tensor,         # (B, N)            long
        task_idx:     torch.Tensor,         # (B,)              long  — integer task indices
        bypass_qwen:  bool = False,         # Stage 1: skip Qwen, pool ev_proj directly
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

        # 1. RMSNorm → MLP projection for events (Qwen3-style: norm before linear)
        ev_normed = self.input_norm(event_embs)                            # (B, N, bert_dim)
        ev_proj   = self.bert_proj_2(F.gelu(self.bert_proj_1(ev_normed))) # (B, N, qwen_dim)

        mask_f = event_mask.float().unsqueeze(-1)   # (B, N, 1)

        # ── Stage-1 bypass: skip Qwen, pool ev_proj directly ──────────────────
        if bypass_qwen:
            pre_emb = (ev_proj * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)  # (B, D)
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
        dtype:            torch.dtype = torch.float32,
    ) -> "DiseaseAwareEHREncoder":
        qwen_lora = PeftModel.from_pretrained(qwen_base, str(save_dir / "lora"))
        encoder   = cls(
            qwen_lora, bert_dim, qwen_dim,
            task_prefix_ids, task_prefix_mask, middle_ids,
            dtype=dtype,
        )
        extra = torch.load(save_dir / "extra_modules.pt", map_location="cpu")
        encoder.bert_proj_1.load_state_dict(extra["bert_proj_1"])
        encoder.bert_proj_2.load_state_dict(extra["bert_proj_2"])
        if "input_norm" in extra:
            encoder.input_norm.load_state_dict(extra["input_norm"])
        return encoder
