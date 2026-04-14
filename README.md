# discord-voice-cloned-messages

## Overview

Python CLI that generates TTS audio from text using [Kyutai's pocket-tts](https://github.com/kyutai-labs/pocket-tts) and sends it as a Discord voice message. Supports both built-in voices and cloned voice embeddings stored as `.safetensors` files.

## Project Structure

```
discord-voice-cloned-messages/
├── speak.py             # TTS generation + Discord voice message sending
├── generate.py          # WAV → .safetensors voice embedding extraction
├── config.toml          # Agent-to-voice mapping (gitignored)
├── tensors/             # Voice embedding files
├── tests/               # pytest tests
├── pyproject.toml       # Dependencies
├── CLAUDE.md            # LLM guidance
└── PLAN.md              # This file
```

## Components

### `speak.py`

CLI: `python speak.py AGENT_NAME CHANNEL_ID`

Reads two lines from stdin (bot token, text), then:

1. Resolve voice from config (see Voice Resolution below)
2. Print agent name, channel, voice, text to stdout (never the token)
3. Generate WAV from text via `TTSModel.generate_audio()` at 24kHz
4. Convert WAV → OGG (Opus, 48kHz, mono, 32kbps) via ffmpeg
5. Extract waveform (256 bytes) and duration from WAV
6. Send OGG as Discord voice message via raw HTTP API with:
   - `flags: 8192` (IS_VOICE_MESSAGE)
   - `waveform`: base64-encoded 256-byte amplitude array
   - `duration_secs`: float
   - `content_type: audio/ogg`
7. Clean up temp files

Uses `discord.http.MultipartParameters` because discord.py's `channel.send()` does not support voice message flags.

### `generate.py`

CLI: `python generate.py PATH_TO_WAV`

1. Validate WAV file exists and has `.wav` extension
2. Create `tensors/` directory if needed
3. Load pocket-tts model, extract voice state from WAV
4. Export voice state to `tensors/<stem>.safetensors` via `export_model_state()`
5. Print output path to stdout

### `config.toml`

```toml
[agents.agent_name]
voice = "voice_identifier"  # optional, defaults to "eponine"
```

### Voice Resolution

`get_voice()` resolves voices in this priority order:

1. **No agent / no voice key** → fall back to `"eponine"`
2. **`tensors/<voice>.safetensors` exists** → return full path to safetensors file
3. **Built-in pocket-tts voice** → return name as-is
4. **Unknown voice** → raise `ClickException`

Built-in voices: `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`.

### 4. Dependencies

| Package | Purpose |
|---------|---------|
| `click` | CLI argument parsing |
| `tomli` | TOML config parsing (Python 3.10 has no stdlib `tomllib`) |
| `pocket-tts` | TTS model for audio generation and voice embedding export |
| `ffmpeg-python` | Pythonic ffmpeg wrapper for WAV→OGG conversion |
| `ffmpeg` | System binary (required by ffmpeg-python) |
| `discord.py` | Discord API client for sending voice messages |
| `pynacl` | NaCl bindings (required by discord.py for full feature support) |
| `davey` | Discord voice gateway support (suppresses discord.py warnings) |
