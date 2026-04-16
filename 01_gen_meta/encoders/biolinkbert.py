from .base import EventEncoder, mean_pool


class BioLinkBERTEncoder(EventEncoder):
    NAME = "biolinkbert"
    MODEL = "michiyasunaga/BioLinkBERT-base"
    ADD_SPECIAL_TOKENS = True
    TEMPLATE_NAME = "biolinkbert_event.j2"

    def get_embeddings(self, model_output, batch_encoding, tokenizer):
        return mean_pool(
            model_output.last_hidden_state,
            batch_encoding["attention_mask"],
            input_ids=batch_encoding["input_ids"],
            special_token_ids=set(tokenizer.all_special_ids),
        )
