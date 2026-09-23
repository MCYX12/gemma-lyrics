#!/usr/bin/env python3

import argparse
import re
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import load


def parse_args():
    parser = argparse.ArgumentParser(description="Generate lyrics with safer sampling.")
    parser.add_argument(
        "--model",
        default=str(Path("model/gemma4_mlx")),
        help="Path to the MLX model directory.",
    )
    parser.add_argument(
        "--adapter-path",
        default=str(Path("adapter/lora")),
        help="Path to the trained LoRA adapter directory.",
    )
    parser.add_argument(
        "--prompt",
        default="写一段雨夜孤独的歌词",
        help="Prompt for lyric generation.",
    )
    parser.add_argument(
        "--modelfile",
        default=str(Path("Modelfile")),
        help="Path to the Ollama Modelfile used as the shared system prompt source.",
    )
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the generated text, without stats or seed output.",
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=0,
        help="Minimum non-empty lyric lines required before accepting a generation.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="How many generation attempts to try when --min-lines is set.",
    )
    parser.add_argument(
        "--structure-template",
        choices=("none", "full_song", "short"),
        default="none",
        help="Optional post-processing template for lyric structure.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed. Omit for varied results across runs.",
    )
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--repetition-context-size", type=int, default=64)
    parser.add_argument("--presence-penalty", type=float, default=0.3)
    parser.add_argument("--presence-context-size", type=int, default=64)
    parser.add_argument("--frequency-penalty", type=float, default=0.2)
    parser.add_argument("--frequency-context-size", type=int, default=64)
    return parser.parse_args()


def load_system_prompt(modelfile_path: str) -> str | None:
    content = Path(modelfile_path).read_text(encoding="utf-8")
    match = re.search(r'SYSTEM\s+"""(.*?)"""', content, re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


def count_nonempty_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def build_prompt(
    tokenizer,
    system_prompt: str | None,
    user_prompt: str,
    assistant_text: str | None = None,
    structure_template: str = "none",
) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    if assistant_text:
        continuation_instruction = "继续把这首歌词补完整，不要重复前文，不要解释，直接续写剩余段落，直到结构完整。"
        if structure_template == "full_song":
            continuation_instruction = (
                "继续把这首歌词补完整，保持五段结构和段落边界，不要重复前文，不要解释，"
                "直接续写缺少的主歌、副歌、桥段或收束。"
            )
        messages.append({"role": "assistant", "content": assistant_text.rstrip()})
        messages.append(
            {
                "role": "user",
                "content": continuation_instruction,
            }
        )
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def normalize_line(line: str) -> str:
    line = re.sub(r"\s+", "", line.strip())
    line = re.sub(r"[，。！？；：、,.!?;:\"'“”‘’（）()《》【】\[\]…-]", "", line)
    return line


def clean_generated_lyrics(text: str) -> str:
    cleaned_lines: list[str] = []
    previous_normalized = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "<start_of_turn>" in line:
            line = line.split("<start_of_turn>", 1)[0].strip()
        if "<end_of_turn>" in line:
            line = line.split("<end_of_turn>", 1)[0].strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        normalized = normalize_line(line)
        if not normalized:
            continue

        if normalized == previous_normalized:
            continue

        if cleaned_lines:
            recent = [normalize_line(item) for item in cleaned_lines[-4:] if item]
            if recent.count(normalized) >= 2:
                continue

        cleaned_lines.append(line)
        previous_normalized = normalized

    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()
    return "\n".join(cleaned_lines)


def split_nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def average_line_length(lines: list[str]) -> float:
    if not lines:
        return 0.0
    return sum(len(normalize_line(line)) for line in lines) / len(lines)


def chorus_score(lines: list[str]) -> tuple[float, int]:
    avg_length = average_line_length(lines)
    pronoun_bonus = sum(1 for line in lines if any(token in line for token in ("你", "我", "夜", "风", "雨")))
    return (avg_length, -pronoun_bonus)


def render_stanzas(stanzas: list[list[str]]) -> str:
    output: list[str] = []
    for index, stanza in enumerate(stanzas):
        output.extend(stanza)
        if index != len(stanzas) - 1:
            output.append("")
    return "\n".join(output)


def dedupe_lines_preserving_order(lines: list[str], max_occurrences: int = 1) -> list[str]:
    counts: dict[str, int] = {}
    cleaned: list[str] = []
    for line in lines:
        normalized = normalize_line(line)
        if not normalized:
            continue
        count = counts.get(normalized, 0)
        if count >= max_occurrences:
            continue
        counts[normalized] = count + 1
        cleaned.append(line)
    return cleaned


def unique_stanzas(stanzas: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    cleaned: list[list[str]] = []
    for stanza in stanzas:
        key = tuple(normalize_line(line) for line in stanza if normalize_line(line))
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(stanza)
    return cleaned


def fill_stanza(lines: list[str], start: int, size: int = 4) -> list[str]:
    stanza = lines[start : start + size]
    return stanza[:size]


def enforce_full_song_structure(text: str) -> str:
    lines = dedupe_lines_preserving_order(split_nonempty_lines(text), max_occurrences=1)
    if len(lines) < 8:
        return text

    target_line_count = 20 if len(lines) >= 20 else 16 if len(lines) >= 16 else 12
    lines = lines[:target_line_count]

    base_stanzas = [lines[index : index + 4] for index in range(0, len(lines), 4)]
    base_stanzas = [stanza for stanza in base_stanzas if len(stanza) >= 2]
    base_stanzas = unique_stanzas(base_stanzas)

    if len(base_stanzas) >= 5:
        verse1 = fill_stanza(base_stanzas[0], 0)
        verse2 = fill_stanza(base_stanzas[1], 0)
        chorus_pool = base_stanzas[2:]
        chorus = min(chorus_pool, key=chorus_score)[:4]
        bridge_pool = [stanza for stanza in chorus_pool if stanza != chorus]
        bridge = (bridge_pool[0] if bridge_pool else base_stanzas[-1])[:4]
        outro_source = base_stanzas[-1][:4]
        outro = outro_source[:]
        if chorus and normalize_line(chorus[0]) not in {normalize_line(line) for line in outro}:
            if len(outro) >= 4:
                outro[-1] = chorus[0]
            else:
                outro.append(chorus[0])
        return render_stanzas(unique_stanzas([verse1, verse2, chorus, bridge, outro]))

    if len(lines) >= 16:
        verse1 = lines[0:4]
        verse2 = lines[4:8]
        chorus = lines[8:12] if len(lines) >= 12 else lines[4:8]
        bridge = lines[12:16] if len(lines) >= 16 else lines[8:12]
        stanzas = unique_stanzas([verse1, verse2, chorus, bridge])
        if len(stanzas) >= 3:
            outro = stanzas[-1][:]
            if chorus and normalize_line(chorus[0]) not in {normalize_line(line) for line in outro}:
                outro = outro[:3] + [chorus[0]]
            stanzas.append(outro)
        return render_stanzas(stanzas)

    return render_stanzas(base_stanzas)


def repeated_line_penalty(text: str) -> int:
    counts: dict[str, int] = {}
    penalty = 0
    for raw_line in text.splitlines():
        normalized = normalize_line(raw_line)
        if not normalized:
            continue
        counts[normalized] = counts.get(normalized, 0) + 1
    for count in counts.values():
        if count > 1:
            penalty += count - 1
    return penalty


def structure_bonus(text: str) -> int:
    lines = [line.strip() for line in text.splitlines()]
    nonempty = [line for line in lines if line]
    blank_count = sum(1 for line in lines if not line)
    bonus = len(nonempty)
    if blank_count >= 3:
        bonus += 4
    elif blank_count >= 1:
        bonus += 2
    return bonus


def score_lyrics(text: str, min_lines: int) -> tuple[int, int, int]:
    line_count = count_nonempty_lines(text)
    penalty = repeated_line_penalty(text)
    shortfall = max(0, min_lines - line_count)
    marker_penalty = text.count("<start_of_turn>") * 10 + text.count("<end_of_turn>") * 10
    stanza_count = len([block for block in text.split("\n\n") if block.strip()])
    stanza_bonus = 0
    if stanza_count >= 5:
        stanza_bonus = 8
    elif stanza_count == 4:
        stanza_bonus = 5
    score = structure_bonus(text) - penalty * 4 - shortfall * 3 - marker_penalty + stanza_bonus
    return score, line_count, penalty


def main():
    args = parse_args()
    model, tokenizer = load(
        args.model,
        adapter_path=args.adapter_path,
        tokenizer_config={"trust_remote_code": True},
    )
    system_prompt = load_system_prompt(args.modelfile)
    quiet_generation = args.quiet or args.min_lines > 0 or args.max_attempts > 1
    sampler = make_sampler(
        temp=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    logits_processors = make_logits_processors(
        repetition_penalty=args.repetition_penalty,
        repetition_context_size=args.repetition_context_size,
        presence_penalty=args.presence_penalty,
        presence_context_size=args.presence_context_size,
        frequency_penalty=args.frequency_penalty,
        frequency_context_size=args.frequency_context_size,
    )

    attempt_count = max(1, args.max_attempts)
    best_text = ""
    best_seed = None
    best_score = None

    for attempt in range(attempt_count):
        seed = args.seed if args.seed is not None else (time.time_ns() + attempt) % (2**32)
        mx.random.seed(seed)
        prompt = args.prompt
        if tokenizer.has_chat_template:
            prompt = build_prompt(
                tokenizer,
                system_prompt,
                args.prompt,
                structure_template=args.structure_template,
            )
        text = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            verbose=not quiet_generation,
        )
        accumulated_text = (text or "").rstrip()
        line_count = count_nonempty_lines(accumulated_text)

        continuation_round = 0
        while tokenizer.has_chat_template and line_count < args.min_lines and continuation_round < 2:
            continuation_round += 1
            seed = args.seed if args.seed is not None else (time.time_ns() + attempt + continuation_round) % (2**32)
            mx.random.seed(seed)
            continuation_prompt = build_prompt(
                tokenizer,
                system_prompt,
                args.prompt,
                assistant_text=accumulated_text,
                structure_template=args.structure_template,
            )
            continuation_text = generate(
                model,
                tokenizer,
                continuation_prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                verbose=False,
            )
            continuation_text = (continuation_text or "").strip()
            if not continuation_text:
                break
            accumulated_text = f"{accumulated_text}\n{continuation_text}".strip()
            line_count = count_nonempty_lines(accumulated_text)

        cleaned_text = clean_generated_lyrics(accumulated_text)
        if args.structure_template == "full_song":
            cleaned_text = enforce_full_song_structure(cleaned_text)
        score, line_count, _penalty = score_lyrics(cleaned_text, args.min_lines)

        if best_score is None or score > best_score:
            best_text = cleaned_text
            best_seed = seed
            best_score = score
        if line_count >= args.min_lines:
            if best_score is not None and best_score >= args.min_lines:
                break

    if quiet_generation:
        if best_text and not best_text.endswith("\n"):
            print(best_text)
        else:
            print(best_text, end="")
        return

    print(f"Seed: {best_seed}")


if __name__ == "__main__":
    main()
