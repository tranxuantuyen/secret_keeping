"""Token position and top-k helpers preserving legacy conventions."""

from __future__ import annotations

from typing import Any


FIRST_PERSON_PRONOUNS = {"i", "me", "my", "mine", "we", "us", "our", "ours"}


def second_start_of_turn(tokens, tokenizer) -> int:
    start_id = tokenizer.encode("<start_of_turn>", add_special_tokens=False)[0]
    positions = [index for index, token_id in enumerate(tokens) if int(token_id) == int(start_id)]
    return positions[1] if len(positions) >= 2 else -1


def first_person_pronoun_positions(tokens, tokenizer, assistant_start: int) -> list[int]:
    limit = assistant_start if assistant_start > 0 else len(tokens)
    positions = []
    for index in range(limit):
        decoded = tokenizer.decode([int(tokens[index])]).strip().lower()
        if decoded in FIRST_PERSON_PRONOUNS:
            positions.append(index)
    return positions


def select_positions(tokens, tokenizer, mode: str) -> tuple[list[int], int]:
    assistant_start = second_start_of_turn(tokens, tokenizer)
    if assistant_start < 0:
        raise ValueError("Second <start_of_turn> token not found in chat-templated sequence")
    if mode in {"single_position", "assistant_start"}:
        positions = [assistant_start]
    elif mode == "control_tokens_average":
        positions = [assistant_start, assistant_start + 1]
    elif mode == "first_person_pronouns":
        positions = first_person_pronoun_positions(tokens, tokenizer, assistant_start)
    elif mode in {"response_tokens", "response_tokens_zeroed"}:
        positions = list(range(assistant_start + 1, len(tokens)))
    elif mode == "prompt_last_token":
        positions = [assistant_start - 1]
    else:
        raise ValueError(f"Unsupported token-position mode: {mode}")
    if not positions:
        raise ValueError(f"No token positions found for mode={mode}")
    if max(positions) >= len(tokens):
        raise ValueError(f"Position {max(positions)} exceeds sequence length {len(tokens)}")
    return positions, assistant_start


def position_metadata(tokens, tokenizer, positions: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "position": int(position),
            "token_id": int(tokens[position]),
            "token": tokenizer.decode([int(tokens[position])]),
        }
        for position in positions
    ]


def top_k_tokens(values, tokenizer, k: int) -> list[dict[str, Any]]:
    import torch

    k = min(k, int(values.shape[-1]))
    top_values, top_indices = torch.topk(values.float(), k=k, largest=True, sorted=True)
    return [
        {
            "rank": rank + 1,
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)]),
            "probability": float(value),
        }
        for rank, (value, token_id) in enumerate(zip(top_values.cpu(), top_indices.cpu()))
    ]

