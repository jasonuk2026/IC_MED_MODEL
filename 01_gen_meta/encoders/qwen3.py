from .base import EventEncoder


class Qwen3Encoder(EventEncoder):
    NAME = "qwen3"
    MODEL = "Qwen/Qwen3-0.6B"
    ADD_SPECIAL_TOKENS = False
    TEMPLATE_NAME = "biolinkbert_event.j2"

    def get_embeddings(self, model_output, batch_encoding, tokenizer):
        return self.pool_hidden(
            model_output.last_hidden_state,
            batch_encoding["attention_mask"],
            input_ids=batch_encoding["input_ids"],
            special_token_ids=None,
        )
