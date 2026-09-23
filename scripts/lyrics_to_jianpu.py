#!/usr/bin/env python3

import argparse
import re
import sys

PUNCTUATION = set(" ，。！？；：、,.!?;:()（）【】[]《》“”\"'…-")

PATTERN_BANKS = {
    "verse": [
        ["5", "5", "6", "1'", "7", "6", "5", "3"],
        ["3", "3", "5", "6", "5", "3", "2", "1"],
        ["2", "3", "5", "5", "6", "5", "3", "2"],
        ["3", "5", "6", "5", "3", "2", "1", "1"],
    ],
    "chorus": [
        ["5", "6", "1'", "1'", "7", "6", "5", "3"],
        ["3", "5", "6", "1'", "2'", "1'", "7", "6"],
        ["6", "6", "5", "3", "5", "6", "1'", "7"],
        ["5", "3", "2", "1", "2", "3", "5", "5"],
    ],
    "bridge": [
        ["6", "5", "3", "5", "6", "7", "1'", "6"],
        ["5", "3", "2", "3", "5", "6", "5", "3"],
        ["2", "2", "3", "5", "6", "5", "3", "2"],
        ["3", "2", "1", "2", "3", "5", "2", "1"],
    ],
    "outro": [
        ["5", "5", "3", "2", "3", "2", "1", "1"],
        ["3", "5", "3", "2", "1", "2", "1", "1"],
        ["2", "3", "2", "1", "2", "1", "7", "1"],
        ["3", "2", "1", "1", "2", "1", "1", "1"],
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert lyrics into deterministic numbered notation.")
    parser.add_argument("--key", default="1=C", help="Key signature header.")
    parser.add_argument("--meter", default="4/4", help="Meter header.")
    parser.add_argument(
        "--lyrics",
        default=None,
        help="Lyrics text to convert. If omitted, reads from stdin.",
    )
    return parser.parse_args()


def clean_lyrics(text: str) -> list[str | None]:
    lines: list[str | None] = []
    for raw_line in text.replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if "<start_of_turn>" in line:
            line = line.split("<start_of_turn>", 1)[0].strip()
        if "<end_of_turn>" in line:
            line = line.split("<end_of_turn>", 1)[0].strip()
        if not line:
            if lines and lines[-1] is not None:
                lines.append(None)
            continue
        if line == "==========":
            continue
        if line.startswith(("Prompt:", "Generation:", "Peak memory:", "Seed:")):
            continue
        line = line.strip("\"")
        if line:
            lines.append(line)
    while lines and lines[-1] is None:
        lines.pop()
    return lines


def normalize_line(line: str) -> str:
    return re.sub(r"[，。！？；：、,.!?;:\"'“”‘’（）()《》【】\[\]…\s-]", "", line.strip())


def dedupe_stanza_lines(stanzas: list[list[str]]) -> list[list[str]]:
    cleaned_stanzas: list[list[str]] = []
    seen_recent: list[str] = []

    for stanza in stanzas:
        cleaned_stanza: list[str] = []
        for line in stanza:
            normalized = normalize_line(line)
            if not normalized:
                continue
            if cleaned_stanza and normalize_line(cleaned_stanza[-1]) == normalized:
                continue
            if seen_recent.count(normalized) >= 2:
                continue
            cleaned_stanza.append(line)
            seen_recent.append(normalized)
            seen_recent = seen_recent[-8:]
        if cleaned_stanza:
            cleaned_stanzas.append(cleaned_stanza)
    return cleaned_stanzas


def group_stanzas(lines: list[str | None]) -> list[list[str]]:
    stanzas: list[list[str]] = [[]]
    for line in lines:
        if line is None:
            if stanzas[-1]:
                stanzas.append([])
            continue
        stanzas[-1].append(line)
    return [stanza for stanza in stanzas if stanza]


def split_syllables(line: str) -> list[str]:
    units: list[str] = []
    index = 0
    while index < len(line):
        char = line[index]
        if char.isspace() or char in PUNCTUATION:
            index += 1
            continue
        if char.isascii() and char.isalnum():
            end = index + 1
            while end < len(line) and line[end].isascii() and line[end].isalnum():
                end += 1
            units.append(line[index:end])
            index = end
            continue
        units.append(char)
        index += 1
    return units


def classify_section(stanza_index: int, stanza_count: int) -> str:
    if stanza_count <= 1:
        return "outro"
    if stanza_index == stanza_count - 1:
        return "outro"
    if stanza_count >= 5 and stanza_index == stanza_count - 2:
        return "bridge"
    if stanza_index >= 2:
        return "chorus"
    return "verse"


def stretch_pattern(pattern: list[str], syllable_count: int) -> list[str]:
    if syllable_count <= 0:
        return ["0", "-", "-", "-"]
    if syllable_count == 1:
        return [pattern[0]]

    notes = []
    pattern_length = len(pattern)
    for position in range(syllable_count):
        index = min(pattern_length - 1, (position * pattern_length) // syllable_count)
        notes.append(pattern[index])
    notes[-1] = pattern[-1]
    return notes


def apply_cadence(notes: list[str], section: str, is_last_line: bool) -> list[str]:
    if not notes:
        return ["0", "-", "-", "-"]
    adjusted = notes[:]
    if section == "chorus":
        adjusted[-1] = "5"
    if section == "bridge":
        adjusted[-1] = "2"
    if section == "outro":
        adjusted[-1] = "1"
    if is_last_line:
        adjusted[-1] = "1"
    return adjusted


def add_bar_holds(notes: list[str]) -> list[str]:
    adjusted = notes[:]
    target_length = max(4, ((len(adjusted) + 3) // 4) * 4)
    while len(adjusted) < target_length:
        adjusted.append("-")
    return adjusted


def format_notes(notes: list[str]) -> str:
    chunks = []
    for index in range(0, len(notes), 4):
        chunks.append(" ".join(notes[index : index + 4]))
    return " | ".join(chunks)


def render_jianpu(lyrics_text: str, key: str, meter: str) -> str:
    lines = clean_lyrics(lyrics_text)
    stanzas = dedupe_stanza_lines(group_stanzas(lines))
    if not stanzas:
        return f"{key} {meter}"

    output_lines = [f"{key} {meter}"]
    global_line_index = 0
    for stanza_index, stanza in enumerate(stanzas):
        section = classify_section(stanza_index, len(stanzas))
        pattern_bank = PATTERN_BANKS[section]
        for line_index, lyric_line in enumerate(stanza):
            syllables = split_syllables(lyric_line)
            pattern = pattern_bank[(global_line_index + line_index) % len(pattern_bank)]
            notes = stretch_pattern(pattern, len(syllables))
            notes = apply_cadence(
                notes,
                section=section,
                is_last_line=stanza_index == len(stanzas) - 1 and line_index == len(stanza) - 1,
            )
            notes = add_bar_holds(notes)
            output_lines.append(format_notes(notes))
            output_lines.append(lyric_line)
        global_line_index += len(stanza)
        if stanza_index != len(stanzas) - 1:
            output_lines.append("")
    return "\n".join(output_lines)


def main():
    args = parse_args()
    lyrics_text = args.lyrics if args.lyrics is not None else sys.stdin.read()
    print(render_jianpu(lyrics_text, key=args.key, meter=args.meter))


if __name__ == "__main__":
    main()
