from .biolinkbert import BioLinkBERTEncoder
from .qwen3 import Qwen3Encoder


ENCODER_REGISTRY = {
    "biolinkbert": BioLinkBERTEncoder,
    "qwen3": Qwen3Encoder,
}


def get_encoder(name, **kwargs):
    key = name.lower()
    if key not in ENCODER_REGISTRY:
        raise ValueError(
            "Unknown encoder {!r}. Available encoders: {}".format(
                name, ", ".join(sorted(ENCODER_REGISTRY))
            )
        )
    return ENCODER_REGISTRY[key](**kwargs)
