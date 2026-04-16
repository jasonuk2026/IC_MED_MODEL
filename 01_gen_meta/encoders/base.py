import os

from jinja2 import Environment, FileSystemLoader, StrictUndefined


class EventEncoder(object):
    NAME = None
    MODEL = None
    ADD_SPECIAL_TOKENS = True
    TEMPLATE_NAME = None

    def __init__(self, model_name=None, template_path=None):
        self.model_name = model_name or self.MODEL
        self.template_path = template_path or self.default_template_path()
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
        return rendered or None

    def get_embeddings(self, model_output, batch_encoding, tokenizer):
        raise NotImplementedError

    def postprocess_embeddings(self, embeddings):
        return embeddings.float()


def mean_pool(
    last_hidden_state,
    attention_mask,
    input_ids=None,
    special_token_ids=None,
):
    import torch

    pool_mask = attention_mask.bool()
    if input_ids is not None and special_token_ids:
        special = torch.zeros_like(attention_mask, dtype=torch.bool)
        for sid in special_token_ids:
            special |= (input_ids == sid)
        pool_mask = pool_mask & ~special

    pool_mask_f = pool_mask.float().unsqueeze(-1)
    sum_emb = (last_hidden_state * pool_mask_f).sum(dim=1)
    count = pool_mask_f.sum(dim=1).clamp(min=1e-9)
    return sum_emb / count
