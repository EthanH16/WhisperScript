#!/usr/bin/env python3
"""
Reusable MP3 transcription + subtitle-axis generator.

Edit MP3_PATH below, or run:
  python transcribe_subtitle_axis.py --mp3 "/path/to/audio.mp3"

Outputs are written to:
  <script_dir>/out/<mp3_stem>_subtitle/
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


# ---- User variables -------------------------------------------------------

# Change this for future runs, or pass --mp3 on the command line.
MP3_PATH = ""

# All generated files go under <script_dir>/out/<mp3_stem>_subtitle/.
OUTPUT_ROOT = "out"

# "small" is faster; "medium" is more accurate for Chinese + English terms.
MODEL_NAME = "medium"

# Optional: keep models next to this script for reuse.
MODEL_DIR = "models"

# Edit this dictionary for your own repeated ASR corrections.
TEXT_REPLACEMENTS = [
    (r"哈喽", "Hello"),
    (r"录点的口播视频", "露脸的口播视频"),
    (r"web\s*coding", "Vibe Coding"),
    (r"webcoding", "Vibe Coding"),
    (r"谷歌的hexon", "Google 的 Hackathon"),
    (r"hexon", "Hackathon"),
    (r"app store", "App Store"),
    (r"APP", "App"),
    (r"可附上的代码", "可复用的代码"),
    (r"无门槛的产生任何代码", "无门槛地生成任何代码"),
    (r"一个真正上限的产品", "一个真正上线的产品"),
]


MODEL_URLS = {
    "small": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
    "medium": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def require_binary(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise SystemExit(
            f"Missing required command: {name}\n"
            f"Install it first. On macOS: brew install whisper-cpp ffmpeg"
        )
    return found


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def model_path(model_name: str) -> Path:
    return script_dir() / MODEL_DIR / f"ggml-{model_name}.bin"


def ensure_model(model_name: str) -> Path:
    path = model_path(model_name)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path

    url = MODEL_URLS.get(model_name)
    if not url:
        raise SystemExit(f"No download URL configured for model: {model_name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Whisper model '{model_name}' to {path}")
    print("This happens once; later runs reuse the local model.")
    urllib.request.urlretrieve(url, path)
    return path


def fmt_ms(ms: int) -> str:
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    seconds, ms = divmod(ms, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def srt_time(ts: str) -> str:
    return ts.replace(".", ",")


def correct_text(text: str) -> str:
    for pattern, replacement in TEXT_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def compact_units(text: str) -> list[str]:
    units: list[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char.isspace():
            i += 1
            continue
        if re.match(r"[A-Za-z0-9_-]", char):
            j = i + 1
            while j < len(text) and re.match(r"[A-Za-z0-9_-]", text[j]):
                j += 1
            units.append(text[i:j])
            i = j
        else:
            units.append(char)
            i += 1
    return units


def raw_char_timeline(segment: dict) -> list[tuple[str, int, int]]:
    chars: list[tuple[str, int, int]] = []
    for token in segment.get("tokens", []):
        text = token.get("text", "")
        if not text or text.startswith("[_") or text.isspace():
            continue
        clean = text.strip()
        if not clean:
            continue

        start = int(token["offsets"]["from"])
        end = int(token["offsets"]["to"])
        duration = max(0, end - start)
        token_chars = [c for c in clean if not c.isspace()]
        if not token_chars:
            continue

        for index, char in enumerate(token_chars):
            char_start = start + round(duration * index / len(token_chars))
            char_end = start + round(duration * (index + 1) / len(token_chars))
            chars.append((char, char_start, char_end))
    return chars


def build_segments(data: dict) -> list[dict]:
    segments = []
    for index, segment in enumerate(data["transcription"], 1):
        start = int(segment["offsets"]["from"])
        end = int(segment["offsets"]["to"])
        raw = segment["text"].strip()
        text = correct_text(raw)
        segments.append(
            {
                "idx": index,
                "start": fmt_ms(start),
                "end": fmt_ms(end),
                "start_ms": start,
                "end_ms": end,
                "text": text,
                "raw": raw,
            }
        )
    return segments


def build_corrected_units(data: dict) -> list[dict]:
    units: list[dict] = []
    unit_index = 1

    for segment_index, segment in enumerate(data["transcription"], 1):
        raw = segment["text"].strip()
        corrected = correct_text(raw)
        start = int(segment["offsets"]["from"])
        end = int(segment["offsets"]["to"])

        raw_chars = raw_char_timeline(segment)
        raw_compact = "".join(char for char, _, __ in raw_chars)
        corrected_compact = "".join(char for char in corrected if not char.isspace())

        matcher = difflib.SequenceMatcher(
            a=raw_compact.lower(),
            b=corrected_compact.lower(),
            autojunk=False,
        )

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if j1 == j2:
                continue
            if i1 < i2 and raw_chars:
                span_start = raw_chars[i1][1]
                span_end = raw_chars[i2 - 1][2]
            else:
                span_start = start if not units else int(units[-1]["end_ms"])
                span_end = span_start
            if span_end <= span_start:
                span_end = span_start + max(
                    20, round((end - start) / max(1, len(corrected_compact)))
                )

            tokens = compact_units(corrected_compact[j1:j2])
            duration = span_end - span_start
            for token_index, token in enumerate(tokens):
                token_start = span_start + round(duration * token_index / len(tokens))
                token_end = span_start + round(duration * (token_index + 1) / len(tokens))
                units.append(
                    {
                        "idx": unit_index,
                        "segment": segment_index,
                        "start": fmt_ms(token_start),
                        "end": fmt_ms(token_end),
                        "start_ms": token_start,
                        "end_ms": token_end,
                        "text": token,
                    }
                )
                unit_index += 1

    return merge_adjacent_english_units(units)


def merge_adjacent_english_units(units: list[dict]) -> list[dict]:
    merged: list[dict] = []
    i = 0
    while i < len(units):
        current = dict(units[i])
        if re.fullmatch(r"[A-Za-z0-9_-]+", current["text"] or ""):
            j = i + 1
            while (
                j < len(units)
                and units[j]["segment"] == current["segment"]
                and re.fullmatch(r"[A-Za-z0-9_-]+", units[j]["text"] or "")
                and int(units[j]["start_ms"]) - int(units[j - 1]["end_ms"]) <= 180
            ):
                current["text"] += units[j]["text"]
                current["end"] = units[j]["end"]
                current["end_ms"] = units[j]["end_ms"]
                j += 1
            merged.append(current)
            i = j
        else:
            merged.append(current)
            i += 1

    for index, unit in enumerate(merged, 1):
        unit["idx"] = index
        unit["text"] = unit["text"].replace("VibeCoding", "Vibe Coding")
    return merged


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_corrected_srt(path: Path, segments: list[dict]) -> None:
    lines: list[str] = []
    for segment in segments:
        lines.extend(
            [
                str(segment["idx"]),
                f"{srt_time(segment['start'])} --> {srt_time(segment['end'])}",
                segment["text"],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(path: Path, mp3_path: Path, segments: list[dict], units: list[dict]) -> None:
    lines: list[str] = []
    lines.append(f"# {mp3_path.name} subtitle axis\n")
    lines.append(f"Source: `{mp3_path}`\n")
    lines.append(
        "Note: generated with whisper.cpp medium/small JSON timestamps. "
        "The word/character axis is aligned to corrected text and is intended "
        "for editing rhythm reference, not frame-perfect manual alignment.\n"
    )

    lines.append("## Sentence Subtitle Axis\n")
    lines.append("| # | Timestamp | Text |\n|---:|---|---|")
    for segment in segments:
        lines.append(
            f"| {segment['idx']} | {segment['start']} --> {segment['end']} | {segment['text']} |"
        )

    lines.append("\n## Raw ASR Sentence Axis\n")
    lines.append("| # | Timestamp | Raw ASR |\n|---:|---|---|")
    for segment in segments:
        lines.append(
            f"| {segment['idx']} | {segment['start']} --> {segment['end']} | {segment['raw']} |"
        )

    lines.append("\n## Corrected Word/Character Rhythm Axis\n")
    lines.append("| # | Segment | Timestamp | Unit |\n|---:|---:|---|---|")
    for unit in units:
        lines.append(
            f"| {unit['idx']} | {unit['segment']} | {unit['start']} --> {unit['end']} | {unit['text']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def output_dir_for(mp3_path: Path, output_root: Path) -> Path:
    return output_root / f"{mp3_path.stem}_subtitle"


def transcribe(mp3_path: Path, out_dir: Path, model: Path) -> Path:
    output_base = out_dir / f"{mp3_path.stem}_medium"
    run(
        [
            "whisper-cli",
            "--no-gpu",
            "-m",
            str(model),
            "-l",
            "zh",
            "-osrt",
            "-ovtt",
            "-oj",
            "-ojf",
            "-owts",
            "-of",
            str(output_base),
            str(mp3_path),
        ]
    )
    return output_base.with_suffix(".json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mp3", default=MP3_PATH, help="Input mp3 path.")
    parser.add_argument("--output-root", default=OUTPUT_ROOT, help="Output root folder.")
    parser.add_argument("--model", default=MODEL_NAME, choices=sorted(MODEL_URLS))
    parser.add_argument(
        "--skip-transcribe",
        action="store_true",
        help="Reuse existing *_medium.json in the output folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_binary("ffmpeg")
    require_binary("ffprobe")
    require_binary("whisper-cli")

    mp3_raw = (args.mp3 or "").strip()
    if not mp3_raw:
        raise SystemExit("MP3 path is empty. Pass --mp3 or set MP3_PATH at the top of the script.")

    mp3_path = Path(mp3_raw).expanduser().resolve()
    if not mp3_path.exists():
        raise SystemExit(f"MP3 not found: {mp3_path}")

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = script_dir() / output_root
    out_dir = output_dir_for(mp3_path, output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ensure_model(args.model)
    json_path = out_dir / f"{mp3_path.stem}_medium.json"
    if args.skip_transcribe and json_path.exists():
        print(f"Reusing existing JSON: {json_path}")
    else:
        json_path = transcribe(mp3_path, out_dir, model)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments = build_segments(data)
    units = build_corrected_units(data)

    write_markdown(out_dir / f"{mp3_path.stem}_subtitle_axis.md", mp3_path, segments, units)
    write_corrected_srt(out_dir / f"{mp3_path.stem}_corrected.srt", segments)
    write_csv(
        out_dir / f"{mp3_path.stem}_sentence_axis.csv",
        segments,
        ["idx", "start", "end", "start_ms", "end_ms", "text", "raw"],
    )
    write_csv(
        out_dir / f"{mp3_path.stem}_corrected_char_word_axis.csv",
        units,
        ["idx", "segment", "start", "end", "start_ms", "end_ms", "text"],
    )

    print("\nDone. Generated files:")
    for file in sorted(out_dir.iterdir()):
        if file.is_file():
            print(" -", file)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
