from __future__ import annotations

import re
from typing import Iterable


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def prepare_source_text(text: str, model_type: str) -> str:
    if model_type in {"mt5", "byt5"}:
        return f"translate English to Swahili: {text}"
    return text


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text.strip())]
    return [paragraph for paragraph in paragraphs if paragraph]


def split_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(text.strip())]
    return [sentence for sentence in sentences if sentence]


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def chunk_text(text: str, tokenizer, model_type: str, max_source_tokens: int) -> list[str]:
    """Split long input without silently dropping later paragraphs."""
    if token_count(tokenizer, prepare_source_text(text, model_type)) <= max_source_tokens:
        return [text.strip()]

    chunks: list[str] = []
    for paragraph in split_paragraphs(text):
        prepared_paragraph = prepare_source_text(paragraph, model_type)
        if token_count(tokenizer, prepared_paragraph) <= max_source_tokens:
            chunks.append(paragraph)
            continue

        current: list[str] = []
        for sentence in split_sentences(paragraph):
            candidate = " ".join([*current, sentence]).strip()
            if current and token_count(tokenizer, prepare_source_text(candidate, model_type)) > max_source_tokens:
                chunks.append(" ".join(current).strip())
                current = [sentence]
            else:
                current.append(sentence)

        if current:
            current_text = " ".join(current).strip()
            if token_count(tokenizer, prepare_source_text(current_text, model_type)) <= max_source_tokens:
                chunks.append(current_text)
            else:
                chunks.extend(split_oversized_sentence(current_text, tokenizer, model_type, max_source_tokens))

    return [chunk for chunk in chunks if chunk]


def split_oversized_sentence(text: str, tokenizer, model_type: str, max_source_tokens: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join([*current, word]).strip()
        if current and token_count(tokenizer, prepare_source_text(candidate, model_type)) > max_source_tokens:
            chunks.append(" ".join(current).strip())
            current = [word]
        else:
            current.append(word)

    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def translate_chunks(
    *,
    chunks: list[str],
    tokenizer,
    model,
    model_type: str,
    device: str,
    batch_size: int,
    num_beams: int,
    max_new_tokens: int,
    length_penalty: float,
    no_repeat_ngram_size: int,
) -> list[str]:
    import torch

    generate_kwargs = {
        "num_beams": num_beams,
        "max_new_tokens": max_new_tokens,
        "length_penalty": length_penalty,
        "early_stopping": True,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }
    if model_type == "nllb":
        generate_kwargs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids("swh_Latn")

    translations: list[str] = []
    for batch in batched(chunks, batch_size):
        source_texts = [prepare_source_text(text, model_type) for text in batch]
        inputs = tokenizer(source_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generate_kwargs)
        translations.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))

    return [translation.strip() for translation in translations]
