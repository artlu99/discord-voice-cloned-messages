import base64
import struct
import tempfile
from pathlib import Path

import click
import discord
import ffmpeg
import numpy as np
import scipy.io.wavfile
import tomli
from discord.http import MultipartParameters
from discord.utils import _to_json
from pocket_tts import TTSModel

CONFIG_PATH = Path(__file__).parent / "config.toml"
TENSORS_DIR = Path(__file__).parent / "tensors"

BUILTIN_VOICES = {"alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"}


def _repair_wav_header(wav_path: Path) -> None:
    with open(wav_path, "r+b") as f:
        file_size = f.seek(0, 2)
        f.seek(4)
        f.write(struct.pack("<I", file_size - 8))
        f.seek(40)
        f.write(struct.pack("<I", file_size - 44))


def _generate_waveform(wav_path: Path) -> tuple[bytes, float]:
    sample_rate, data = scipy.io.wavfile.read(str(wav_path))
    duration_secs = len(data) / sample_rate
    samples = np.abs(data.astype(np.float32))
    if len(samples) == 0:
        return b"\x00" * 256, 0.0
    indices = np.linspace(0, len(samples) - 1, 256, dtype=int)
    waveform = samples[indices]
    peak = waveform.max()
    if peak > 0:
        waveform = waveform / peak * 255
    return bytes(waveform.astype(np.uint8)), duration_secs


def get_voice(
    agent_name: str,
    config_path: Path = CONFIG_PATH,
) -> str:
    with open(config_path, "rb") as f:
        config = tomli.load(f)

    agents = config.get("agents", {})

    voice = agents.get(agent_name, {}).get("voice", "eponine")

    safetensors_path = TENSORS_DIR / f"{voice}.safetensors"
    if safetensors_path.exists():
        return str(safetensors_path)

    if voice in BUILTIN_VOICES:
        return voice

    raise click.ClickException(f"Unknown voice '{voice}'")


def generate_wav(text: str, voice: str, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_path = output_dir / "output.wav"
    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt(voice)
    audio = model.generate_audio(voice_state, text)
    scipy.io.wavfile.write(str(output_path), model.sample_rate, audio.numpy())
    return output_path


def convert_to_ogg(wav_path: str, bitrate: str = "32k") -> Path:
    ogg_path = Path(wav_path).with_suffix(".ogg")
    (
        ffmpeg.input(wav_path)
        .output(str(ogg_path), ac=1, ar=48000, acodec="libopus", audio_bitrate=bitrate)
        .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
    )
    return ogg_path


@click.command()
@click.argument("agent_name")
@click.argument("channel_id")
def speak(agent_name: str, channel_id: str) -> None:
    token = input()
    text = input()

    voice = get_voice(agent_name)
    click.echo(f"Agent: {agent_name}\nChannel: {channel_id}\nVoice: {voice}\nText: {text}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = generate_wav(text, voice, tmp_dir)
        ogg_path = convert_to_ogg(str(wav_path))
        waveform, duration_secs = _generate_waveform(wav_path)
        send_voice_message(token, channel_id, ogg_path, waveform, duration_secs)


def send_voice_message(token: str, channel_id: str, ogg_path: Path, waveform: bytes, duration_secs: float) -> None:
    intents = discord.Intents.default()
    bot = discord.Client(intents=intents)

    @bot.event
    async def on_ready() -> None:
        payload = {
            "flags": 8192,  # IS_VOICE_MESSAGE
            "attachments": [{"id": 0, "filename": "voice_message.ogg", "waveform": base64.b64encode(waveform).decode(), "duration_secs": duration_secs}],
        }

        file = discord.File(ogg_path, filename="voice_message.ogg")

        multipart = [
            {"name": "payload_json", "value": _to_json(payload)},
            {
                "name": "files[0]",
                "value": file.fp,
                "filename": file.filename,
                "content_type": "audio/ogg",
            },
        ]

        params = MultipartParameters(payload=None, multipart=multipart, files=[file])
        await bot.http.send_message(int(channel_id), params=params)
        await bot.close()

    bot.run(token)


if __name__ == "__main__":
    speak()
