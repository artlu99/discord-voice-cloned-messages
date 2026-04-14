from pathlib import Path

import click
from pocket_tts import TTSModel, export_model_state

TENSORS_DIR = Path(__file__).parent / "tensors"


def generate_tensor(wav_path: str, tensors_dir: Path = TENSORS_DIR) -> str:
    wav_file = Path(wav_path)

    if not wav_file.exists():
        raise FileNotFoundError(f"File not found: {wav_path}")

    if wav_file.suffix.lower() != ".wav":
        raise click.ClickException(f"File is not a WAV file: {wav_path}")

    tensors_dir.mkdir(parents=True, exist_ok=True)

    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt(str(wav_file))

    output_name = wav_file.stem + ".safetensors"
    output_path = tensors_dir / output_name
    export_model_state(voice_state, str(output_path))

    return str(output_path)


@click.command()
@click.argument("wav_path", type=click.Path(exists=True))
def generate(wav_path: str) -> None:
    click.echo(generate_tensor(wav_path))


if __name__ == "__main__":
    generate()
