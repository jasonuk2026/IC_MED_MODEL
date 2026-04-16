from .base import EventEncoder, mean_pool


class Qwen3Encoder(EventEncoder):
    NAME = "qwen3"
    MODEL = "Qwen/Qwen3-0.6B"
    ADD_SPECIAL_TOKENS = False
    TEMPLATE_NAME = "biolinkbert_event.j2"

    def get_embeddings(self, model_output, batch_encoding, tokenizer):
        return mean_pool(
            model_output.last_hidden_state,
            batch_encoding["attention_mask"],
            input_ids=None,
            special_token_ids=None,
        )
