import os

from jinja2 import Environment, FileSystemLoader, StrictUndefined
import torch


class EventEncoder(object):
    NAME = None
    MODEL = None
    ADD_SPECIAL_TOKENS = True
    TEMPLATE_NAME = None

    def __init__(
        self,
        model_name=None,
        template_path=None,
        append_token_text=None,
        append_token_name=None,
        pool_max_tokens=None,
        pooling_mode="mean",
    ):
        self.model_name = model_name or self.MODEL
        self.template_path = template_path or self.default_template_path()
        self.append_token_text = append_token_text.strip() if append_token_text else None
        self.append_token_name = append_token_name.strip() if append_token_name else None
        self.pool_max_tokens = pool_max_tokens
        self.pooling_mode = pooling_mode
        self.append_token_id = None
        self._template = self._load_template(self.template_path)

    def default_template_path(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(base_dir, "templates", self.TEMPLATE_NAME)

    def _load_template(self, template_path):
        template_dir = os.path.dirname(template_path)
        template_name = os.path.basename(template_path)
        env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )
        return env.get_template(template_name)

    def build_template_context(
        self,
        code,
        description,
        value,
        unit,
        omop_table="",
        event_type="",
    ):
        return {
            "omop_table": omop_table or "",
            "event_type": event_type or "",
            "code": code or "",
            "description": description or "",
            "value": value or "",
            "unit": unit or "",
        }

    def format_event_text(
        self,
        code,
        description,
        value,
        unit,
        omop_table="",
        event_type="",
    ):
        rendered = self._template.render(
            **self.build_template_context(
                code,
                description,
                value,
                unit,
                omop_table=omop_table,
                event_type=event_type,
            )
        ).strip()
        if not rendered:
            return None
        if self.append_token_text:
            return "{}{}".format(rendered, self.append_token_text)
        return rendered

    def resolve_append_token(self, tokenizer):
        if self.append_token_name:
            token_text = getattr(tokenizer, self.append_token_name, None)
            if not token_text:
                raise ValueError(
                    "Tokenizer {} has no usable {} value".format(
                        tokenizer.__class__.__name__,
                        self.append_token_name,
                    )
                )
            self.append_token_text = token_text

    def _set_append_token_id_from_existing_vocab(self, tokenizer):
        if not self.append_token_text:
            self.append_token_id = None
            return False
        token_id = tokenizer.convert_tokens_to_ids(self.append_token_text)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if token_id is None or (unk_id is not None and token_id == unk_id and self.append_token_text != tokenizer.unk_token):
            self.append_token_id = None
            return False
        self.append_token_id = int(token_id)
        return True

    def configure_tokenizer_and_model(self, tokenizer, model):
        self.resolve_append_token(tokenizer)
        if self.append_token_text:
            if not self._set_append_token_id_from_existing_vocab(tokenizer):
                num_added = tokenizer.add_special_tokens(
                    {"additional_special_tokens": [self.append_token_text]}
                )
                self.append_token_id = tokenizer.convert_tokens_to_ids(self.append_token_text)
                if num_added > 0:
                    model.resize_token_embeddings(len(tokenizer))
        return tokenizer, model

    def build_pool_mask(self, attention_mask, input_ids=None, special_token_ids=None):
        return build_pool_mask(
            attention_mask,
            input_ids=input_ids,
            special_token_ids=special_token_ids,
            suffix_token_id=self.append_token_id,
            max_tokens=self.pool_max_tokens,
            pooling_mode=self.pooling_mode,
        )

    def pool_hidden(self, last_hidden_state, attention_mask, input_ids=None, special_token_ids=None):
        pool_mask = self.build_pool_mask(
            attention_mask,
            input_ids=input_ids,
            special_token_ids=special_token_ids,
        )
        return mean_pool(last_hidden_state, pool_mask)

    def get_embeddings(self, model_output, batch_encoding, tokenizer):
        raise NotImplementedError

    def postprocess_embeddings(self, embeddings):
        return embeddings.float()


def build_pool_mask(
    attention_mask,
    input_ids=None,
    special_token_ids=None,
    suffix_token_id=None,
    max_tokens=None,
    pooling_mode="mean",
):
    if pooling_mode not in {"mean", "suffix_only"}:
        raise ValueError("Unknown pooling_mode={!r}".format(pooling_mode))

    if pooling_mode == "suffix_only":
        if input_ids is None:
            raise ValueError("suffix_only pooling requires input_ids")
        if suffix_token_id is None:
            raise ValueError("suffix_only pooling requires append_token_text or append_token_name")
        valid_lengths = attention_mask.long().sum(dim=1) - 1
        valid_lengths = valid_lengths.clamp(min=0)
        row_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        pool_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        suffix_matches = input_ids[row_idx, valid_lengths] == suffix_token_id
        if not bool(suffix_matches.all()):
            raise ValueError("suffix_only pooling expected the final valid token to be the configured append token")
        pool_mask[row_idx, valid_lengths] = True
        return pool_mask

    pool_mask = attention_mask.bool()
    if input_ids is not None and special_token_ids:
        special = torch.zeros_like(attention_mask, dtype=torch.bool)
        for sid in special_token_ids:
            special |= (input_ids == sid)
        pool_mask = pool_mask & ~special

    if input_ids is not None and suffix_token_id is not None:
        valid_lengths = attention_mask.long().sum(dim=1) - 1
        valid_lengths = valid_lengths.clamp(min=0)
        row_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        suffix_matches = input_ids[row_idx, valid_lengths] == suffix_token_id
        pool_mask[row_idx[suffix_matches], valid_lengths[suffix_matches]] = False

    if max_tokens is not None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive, got {}".format(max_tokens))
        token_ord = torch.cumsum(pool_mask.long(), dim=1)
        pool_mask = pool_mask & (token_ord <= max_tokens)

    return pool_mask


def mean_pool(last_hidden_state, pool_mask):
    pool_mask_f = pool_mask.float().unsqueeze(-1)
    sum_emb = (last_hidden_state * pool_mask_f).sum(dim=1)
    count = pool_mask_f.sum(dim=1).clamp(min=1e-9)
    return sum_emb / count
