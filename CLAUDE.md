# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

Python CLI that generates TTS audio via pocket-tts and sends it as a Discord voice message to a text channel. Voice embeddings can be cloned from WAV files into reusable `.safetensors` files.

See `PLAN.md` for the full specification.

## Tooling

- **Package manager**: `uv` (lock file: `uv.lock`)
- **Python**: 3.10 (`.python-version`)
- **Install dependencies**: `uv sync`
- **Install dev dependencies**: `uv sync --group dev`
- **Run scripts**: `uv run python speak.py`, `uv run python generate.py`
- **Run tests**: `uv run pytest`

## Architecture

### `speak.py` — TTS + Discord voice message sender

CLI: `uv run python speak.py AGENT_NAME CHANNEL_ID`

Reads bot token and text from stdin (two lines). Resolves the agent's voice from `config.toml`, generates WAV audio via pocket-tts, converts to OGG/Opus via ffmpeg, and sends as a Discord voice message using the raw HTTP API with `MultipartParameters` (required for `flags: 8192` IS_VOICE_MESSAGE).

Key functions:
- `get_voice(agent_name)` — resolves voice from config (safetensors file, built-in name, or error)
- `generate_wav(text, voice, output_dir)` — pocket-tts audio generation
- `convert_to_ogg(wav_path)` — ffmpeg WAV→OGG conversion (48kHz mono 32kbps)
- `_generate_waveform(wav_path)` — returns `(base64-waveform-bytes, duration-secs)` tuple
- `send_voice_message(token, channel_id, ogg_path, waveform, duration_secs)` — Discord HTTP upload

### `generate.py` — Voice embedding extractor

CLI: `uv run python generate.py PATH_TO_WAV`

Takes a WAV file, produces a `.safetensors` voice embedding in `tensors/` via `export_model_state()`.

### `config.toml` — Agent configuration

Maps agent names to optional voice identifiers under `[agents.<name>]`. Voice resolution priority:
1. `tensors/<voice>.safetensors` file exists → use that path
2. Voice is a built-in pocket-tts name → use as-is
3. Unknown voice → raise `ClickException`
4. No `voice` key or unknown agent → fall back to `"eponine"`

Built-in voices: `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`.

### `tests/` — pytest

- `test_speak.py` — voice resolution, CLI invocation, WAV generation, OGG conversion, waveform generation (17 tests)
- `test_generate.py` — WAV validation, tensor export (5 tests, pocket-tts mocked)

## Key Decisions

- Python 3.10: use `tomli` instead of stdlib `tomllib` (3.11+)
- `scipy.io.wavfile` for WAV writing (not torchaudio)
- discord.py's `channel.send()` doesn't support voice message flags; must use raw `bot.http.send_message()` with `MultipartParameters`
- Voice message metadata: `flags: 8192`, `waveform` as base64 string, `duration_secs` as float
- The bot token read from stdin must never be printed or logged
- All imports are at module level (no local imports)
