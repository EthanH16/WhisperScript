# WhisperScript

Reusable MP3 transcription and subtitle-axis generator.

This folder is meant to be shared by all TikTok projects.

## One-time Setup

```bash
cd WhisperScript
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You also need the system tools:

```bash
brew install whisper-cpp ffmpeg
```

## CLI Usage

The script supports both a default path variable and an explicit CLI argument.

```bash
source .venv/bin/activate
python transcribe_subtitle_axis.py --mp3 "/path/to/audio.mp3"
```

Or edit `MP3_PATH` at the top of `transcribe_subtitle_axis.py` and run:

```bash
source .venv/bin/activate
python transcribe_subtitle_axis.py
```

Optional flags:

```bash
python transcribe_subtitle_axis.py --mp3 "/path/to/audio.mp3" --model medium
python transcribe_subtitle_axis.py --mp3 "/path/to/audio.mp3" --model small
python transcribe_subtitle_axis.py --mp3 "/path/to/audio.mp3" --output-root out
python transcribe_subtitle_axis.py --mp3 "/path/to/audio.mp3" --skip-transcribe
```

## Output

All generated files go to:

```text
./out/<mp3文件名>_subtitle/
```

Each run writes:

```text
<mp3文件名>_subtitle_axis.md
<mp3文件名>_corrected.srt
<mp3文件名>_sentence_axis.csv
<mp3文件名>_corrected_char_word_axis.csv
<mp3文件名>_medium.json
<mp3文件名>_medium.srt
<mp3文件名>_medium.vtt
```

## Model Cache

The first run downloads the Whisper model into:

```text
./models/
```

Later runs reuse the cached model.
