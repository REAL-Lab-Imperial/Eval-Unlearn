"""Auto-computed CLIP concept-direction vectors for Ring-A-Bell.

Ring-A-Bell's genetic-algorithm prompt discovery needs a concept vector: a
``(seq_len, hidden_dim)`` array of CLIP text-encoder token embeddings
representing a target concept's direction in embedding space (see the
bundled nudity vector — computed offline by the original paper's authors and
shipped in ``data/nudity_vector.npy``). For any concept without a bundled or
user-supplied vector, this module computes one from scratch, following the
paper's differencing approach: the average difference between the token
embeddings of paired "concept-present" and "concept-absent" prompt
templates.
"""
import numpy as np
import torch

_POSITIVE_TEMPLATES = [
    "a photo of {c}",
    "an image of {c}",
    "a picture of {c}",
    "a realistic depiction of {c}",
    "a detailed photo of {c}",
    "a scene showing {c}",
    "an artwork depicting {c}",
    "a high resolution photo of {c}",
]

_NEGATIVE_TEMPLATES = [
    "a photo",
    "an image",
    "a picture",
    "a realistic depiction",
    "a detailed photo",
    "a scene",
    "an artwork",
    "a high resolution photo",
]


@torch.no_grad()
def compute_concept_vector(
    concept: str,
    clip_model,
    clip_processor,
    device: str,
    max_length: int = 77,
) -> np.ndarray:
    """
    Compute a CLIP token-embedding concept-direction vector for ``concept``.

    Averages, over paired "concept-present" vs "concept-absent" prompt
    templates, the difference between their CLIP text-encoder token
    embeddings (``last_hidden_state``, before pooling). Returns a
    ``(max_length, hidden_dim)`` float32 array in the same format as the
    bundled nudity concept vector.
    """
    diffs = []
    for pos_template, neg_template in zip(_POSITIVE_TEMPLATES, _NEGATIVE_TEMPLATES):
        pos_hidden = _encode_tokens(
            pos_template.format(c=concept), clip_model, clip_processor, device, max_length
        )
        neg_hidden = _encode_tokens(
            neg_template, clip_model, clip_processor, device, max_length
        )
        diffs.append((pos_hidden - neg_hidden).cpu().numpy())

    return np.mean(np.stack(diffs, axis=0), axis=0).astype(np.float32)


def _encode_tokens(
    text: str, clip_model, clip_processor, device: str, max_length: int
) -> torch.Tensor:
    """Return the (max_length, hidden_dim) CLIP text-encoder token embeddings for one string."""
    inputs = clip_processor(
        text=[text],
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    outputs = clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state.squeeze(0)
