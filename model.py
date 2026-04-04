"""
model.py

DiseaseAwareEHREncoder — Qwen3-style disease-conditioned patient encoder.

Architecture per sample:
  event_embs   (B, N, 768)  ──► input_norm ──► bert_proj_1(768→768) ──► GELU ──► bert_proj_2(768→D) ──► ev_proj   (B, N, D)
  disease_embs (B, 768)     ──► input_norm ──► bert_proj_1(768→768) ──► GELU ──► bert_proj_2(768→D) ──► dis_token (B, 1, D)

  Qwen input: [prefix | dis_token | middle | ev_proj | EOS]
  Pool: last token (EOS) → L2 normalise

Norm philosophy (Qwen3 style): normalise *before* every neural-network layer,
not after.  input_norm (RMSNorm over bert_dim=768) is applied to the raw
BioLinkBERT embeddings before they enter the projection MLP.  No norm is
applied after any linear layer.
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
    """Qwen model conditioned on disease via a single projected virtual token.

    The BioLinkBERT disease embedding is projected to Qwen's hidden size and
    inserted at the position of the disease name in the prompt template:

        [prefix_embeds | disease_virtual_token | middle_embeds | projected event embs | EOS]

    where prefix = PROMPT_PREFIX = "Please predict disease "
    and   middle = PROMPT_MIDDLE = " based on the following events.\\nStart of medical events:"

    Norm order (Qwen3 style):
        input_norm (RMSNorm over bert_dim) is applied to event_embs and
        disease_embs *before* the projection MLP.  No normalisation is applied
        after any linear layer.

    Trainable modules beyond Qwen LoRA:
        input_norm   RMSNorm(bert_dim)                       (stabilises MLP input)
        bert_proj_1  Linear(bert_dim → bert_dim)  no bias    (hidden)
        bert_proj_2  Linear(bert_dim → qwen_dim)  no bias    (up-project; shared by events and disease)
    """

    def __init__(
        self,
        qwen_model: nn.Module,
        bert_dim:   int,
        qwen_dim:   int,
        prefix_ids: torch.Tensor,               # (1, P1)  token ids for PROMPT_PREFIX
        middle_ids: torch.Tensor,               # (1, P2)  token ids for PROMPT_MIDDLE
        dtype:      torch.dtype = torch.float32,
    ):
        super().__init__()
        self.qwen        = qwen_model
        self.dtype       = dtype
        self.input_norm  = nn.RMSNorm(bert_dim).to(dtype)
        self.bert_proj_1 = nn.Linear(bert_dim, bert_dim, bias=False).to(dtype)
        self.bert_proj_2 = nn.Linear(bert_dim, qwen_dim, bias=False).to(dtype)

        nn.init.xavier_uniform_(self.bert_proj_1.weight)
        nn.init.xavier_uniform_(self.bert_proj_2.weight)

        # Pre-compute frozen text embeddings once; store as buffers so they
        # move to the right device automatically and never appear in the forward graph.
        embed_fn = qwen_model.get_input_embeddings()
        with torch.no_grad():
            self.register_buffer("prefix_embeds", embed_fn(prefix_ids).to(dtype))  # (1, P1, D)
            self.register_buffer("middle_embeds", embed_fn(middle_ids).to(dtype))  # (1, P2, D)
            eos_ids = torch.full((1, 1), EOS_TOKEN_ID, dtype=torch.long)
            self.register_buffer("eos_embed",    embed_fn(eos_ids).to(dtype))       # (1,  1, D)

    def forward(
        self,
        event_embs:   torch.Tensor,   # (B, N, bert_dim)
        event_mask:   torch.Tensor,   # (B, N)            long
        disease_embs: torch.Tensor,   # (B, bert_dim)
    ) -> torch.Tensor:                # (B, qwen_dim)  L2-normalised embeddings
        B      = event_embs.size(0)
        device = event_embs.device

        # Cast inputs to the module's compute dtype
        event_embs   = event_embs.to(self.dtype)
        disease_embs = disease_embs.to(self.dtype)

        # 1. RMSNorm before MLP (Qwen3 style: norm before any linear layer)
        ev_normed  = self.input_norm(event_embs)    # (B, N, bert_dim)
        dis_normed = self.input_norm(disease_embs)  # (B, bert_dim)

        # 2. Project: RMSNorm → Linear → GELU → Linear
        ev_proj   = self.bert_proj_2(F.gelu(self.bert_proj_1(ev_normed)))             # (B, N, qwen_dim)
        dis_token = self.bert_proj_2(F.gelu(self.bert_proj_1(dis_normed))).unsqueeze(1)  # (B, 1, qwen_dim)

        # 3. Retrieve pre-computed text embeddings (registered buffers, frozen).
        #    Sequence layout matches the prompt template:
        #      "Please predict disease <DISEASE> based on the following events.
        #       Start of medical events: <EVENTS>"
        prefix = self.prefix_embeds.expand(B, -1, -1)  # (B, P1, D)
        middle = self.middle_embeds.expand(B, -1, -1)  # (B, P2, D)
        eos    = self.eos_embed.expand(B, -1, -1)       # (B,  1, D)

        # 4. Assemble: [prefix | disease_token | middle | events | EOS]
        inputs_embeds = torch.cat([prefix, dis_token, middle, ev_proj, eos], dim=1)

        P1 = prefix.size(1)
        P2 = middle.size(1)
        ones = lambda n: torch.ones(B, n, dtype=torch.long, device=device)
        attention_mask = torch.cat(
            [ones(P1), ones(1), ones(P2), event_mask, ones(1)], dim=1
        )

        # 5. Forward through Qwen (inputs_embeds bypasses embed_tokens)
        out = self.qwen(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        # 6. Last-token pool (EOS) → L2 normalise
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx   = torch.arange(B, device=device)
        emb = out.last_hidden_state[batch_idx, seq_lengths]
        return F.normalize(emb.float(), p=2, dim=-1)

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def save_checkpoint(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        raw_qwen = self.qwen.module if isinstance(self.qwen, DDP) else self.qwen
        raw_qwen.save_pretrained(str(save_dir / "lora"))
        torch.save(
            {"bert_proj_1": self.bert_proj_1.state_dict(),
             "bert_proj_2": self.bert_proj_2.state_dict(),
             "input_norm":  self.input_norm.state_dict()},
            save_dir / "extra_modules.pt",
        )
        logger.info(f"  Saved checkpoint → {save_dir}")

    @classmethod
    def load_from_checkpoint(
        cls,
        save_dir:   Path,
        qwen_base:  nn.Module,
        bert_dim:   int,
        qwen_dim:   int,
        prefix_ids: torch.Tensor,
        middle_ids: torch.Tensor,
        dtype:      torch.dtype = torch.float32,
    ) -> "DiseaseAwareEHREncoder":
        qwen_lora = PeftModel.from_pretrained(qwen_base, str(save_dir / "lora"))
        encoder   = cls(qwen_lora, bert_dim, qwen_dim, prefix_ids, middle_ids, dtype=dtype)
        extra     = torch.load(save_dir / "extra_modules.pt", map_location="cpu")
        encoder.bert_proj_1.load_state_dict(extra["bert_proj_1"])
        encoder.bert_proj_2.load_state_dict(extra["bert_proj_2"])
        if "input_norm" in extra:
            encoder.input_norm.load_state_dict(extra["input_norm"])
        return encoder
