"""
verify_lora_stage1.py

Verify that:
1. lora_B matrices are all zero at init → LoRA adds no delta to base weights.
2. Base model and LoRA-wrapped model produce identical outputs.
3. After one backward step through the LoRA model (with LoRA frozen),
   lora_B stays zero and bert_proj grads are non-zero.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from model import DiseaseAwareEHREncoder, EOS_TOKEN_ID

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BERT_DIM   = 768

def _set_lora_trainable(model, trainable):
    n = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad_(trainable)
            n += 1
    return n

print("=" * 60)
print("Loading base model …")
base = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True, dtype=torch.float32)
tok  = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ── Test 1: lora_B is zero at init ────────────────────────────────────────────
print("\n[Test 1] lora_B norms at initialization")
lora_cfg = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=8, lora_alpha=16, lora_dropout=0.0,   # dropout=0 for determinism
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
lora_model = get_peft_model(base, lora_cfg)

b_norms = {n: p.norm().item() for n, p in lora_model.named_parameters() if "lora_B" in n}
all_zero = all(v == 0.0 for v in b_norms.values())
print(f"  Number of lora_B tensors : {len(b_norms)}")
print(f"  All lora_B zero          : {all_zero}")
if not all_zero:
    nonzero = [(n, v) for n, v in b_norms.items() if v != 0.0]
    for n, v in nonzero[:5]:
        print(f"    NONZERO: {n} = {v:.6f}")

# ── Test 2: base == lora model output (no dropout, eval mode) ─────────────────
print("\n[Test 2] Forward pass: base vs lora-wrapped (eval mode)")
dummy_ids  = torch.randint(100, 1000, (2, 16))
dummy_mask = torch.ones(2, 16, dtype=torch.long)

base.eval()
lora_model.eval()

with torch.no_grad():
    out_base = base(input_ids=dummy_ids, attention_mask=dummy_mask).last_hidden_state
    out_lora = lora_model(input_ids=dummy_ids, attention_mask=dummy_mask).last_hidden_state

diff = (out_base - out_lora).abs()
print(f"  max  |base - lora| : {diff.max().item():.2e}")
print(f"  mean |base - lora| : {diff.mean().item():.2e}")
print(f"  allclose (atol=1e-6): {torch.allclose(out_base, out_lora, atol=1e-6)}")

# ── Test 3: full DiseaseAwareEHREncoder — stage 1 frozen LoRA backward ────────
print("\n[Test 3] DiseaseAwareEHREncoder stage-1 backward")
print("  (LoRA frozen, only bert_proj should get gradients)\n")

from transformers import AutoTokenizer
PROMPT_PREFIX = "Please predict disease "
PROMPT_MIDDLE = " based on the following events.\nStart of medical events:"
TASK_2_DISEASE_NAME = {
    "new_hypertension":   "hypertension",
    "new_hyperlipidemia": "hyperlipidemia",
    "new_pancan":         "pancreatic cancer",
    "new_celiac":         "celiac disease",
    "new_lupus":          "systemic lupus erythematosus",
    "new_acutemi":        "acute myocardial infarction",
}
TASK_2_IDX = {t: i for i, t in enumerate(sorted(TASK_2_DISEASE_NAME))}

tasks_sorted = sorted(TASK_2_DISEASE_NAME)
pad_id = tok.pad_token_id
all_ids = [tok(PROMPT_PREFIX + TASK_2_DISEASE_NAME[t], add_special_tokens=False)["input_ids"]
           for t in tasks_sorted]
max_len = max(len(ids) for ids in all_ids)
task_prefix_ids  = torch.full((len(tasks_sorted), max_len), pad_id, dtype=torch.long)
task_prefix_mask = torch.zeros((len(tasks_sorted), max_len), dtype=torch.long)
for i, ids in enumerate(all_ids):
    n = len(ids)
    task_prefix_ids[i,  max_len - n:] = torch.tensor(ids)
    task_prefix_mask[i, max_len - n:] = 1

middle_ids = tok(PROMPT_MIDDLE, add_special_tokens=False, return_tensors="pt")["input_ids"]

# fresh lora model for the encoder
base2 = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True, dtype=torch.float32)
base2.config.use_cache = False
lora2 = get_peft_model(base2, lora_cfg)

encoder = DiseaseAwareEHREncoder(
    qwen_model=lora2,
    bert_dim=BERT_DIM,
    qwen_dim=base2.config.hidden_size,
    task_prefix_ids=task_prefix_ids,
    task_prefix_mask=task_prefix_mask,
    middle_ids=middle_ids,
)

# Freeze LoRA as in stage 1
n_frozen = _set_lora_trainable(encoder, False)
print(f"  Frozen {n_frozen} LoRA parameter tensors")

# Confirm what is trainable
trainable = [(n, p.shape) for n, p in encoder.named_parameters() if p.requires_grad]
print(f"  Trainable params ({len(trainable)} tensors):")
for n, s in trainable:
    print(f"    {n}  {list(s)}")

# dummy batch: B=4, N=10 events, BERT_DIM
B, N = 4, 10
event_embs = torch.randn(B, N, BERT_DIM)
event_mask = torch.ones(B, N, dtype=torch.long)
task_idxs  = torch.zeros(B, dtype=torch.long)   # all task 0
labels     = torch.tensor([1, 0, 1, 0], dtype=torch.long)

encoder.train()
emb = encoder(event_embs, event_mask, task_idxs)   # full Qwen forward

# simple triplet-style loss
pos = emb[labels == 1]
neg = emb[labels == 0]
loss = (1.0 - (pos @ neg.T)).mean()
loss.backward()

print(f"\n  Loss value: {loss.item():.4f}")

# Check gradients
print("\n  Gradient norms after backward:")
for n, p in encoder.named_parameters():
    if p.grad is not None:
        gn = p.grad.norm().item()
        print(f"    {n:60s}  grad_norm={gn:.4e}")
    else:
        # only report lora_ params that stayed None
        if "lora_" in n:
            print(f"    {n:60s}  grad=None (frozen ✓)")

# Verify lora_B still zero
b_norms_after = {n: p.norm().item() for n, p in encoder.named_parameters() if "lora_B" in n}
still_zero = all(v == 0.0 for v in b_norms_after.values())
print(f"\n  lora_B still all-zero after backward: {still_zero}")

print("\n" + "=" * 60)
print("All tests done.")
