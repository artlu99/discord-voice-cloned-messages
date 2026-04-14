from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from generate import generate, generate_tensor


class TestGenerateTensor:
    def test_rejects_nonexistent_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            generate_tensor(str(tmp_path / "missing.wav"), tensors_dir=tmp_path / "tensors")

    def test_rejects_non_wav_file(self, tmp_path: Path) -> None:
        non_wav = tmp_path / "audio.mp3"
        non_wav.write_text("not a wav")

        with pytest.raises(Exception, match="not a WAV file"):
            generate_tensor(str(non_wav), tensors_dir=tmp_path / "tensors")

    def test_creates_tensors_directory(self, tmp_path: Path) -> None:
        tensors_dir = tmp_path / "tensors"
        wav_file = tmp_path / "sample.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 100)

        mock_model = MagicMock()
        mock_model.get_state_for_audio_prompt.return_value = MagicMock()

        with patch("generate.TTSModel.load_model", return_value=mock_model), \
             patch("generate.export_model_state"):
            generate_tensor(str(wav_file), tensors_dir=tensors_dir)

        assert tensors_dir.is_dir()

    def test_calls_export_with_correct_path(self, tmp_path: Path) -> None:
        tensors_dir = tmp_path / "tensors"
        wav_file = tmp_path / "my_voice.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 100)

        mock_model = MagicMock()
        voice_state = MagicMock()
        mock_model.get_state_for_audio_prompt.return_value = voice_state

        with patch("generate.TTSModel.load_model", return_value=mock_model), \
             patch("generate.export_model_state") as mock_export:
            result = generate_tensor(str(wav_file), tensors_dir=tensors_dir)

        expected_path = str(tensors_dir / "my_voice.safetensors")
        assert result == expected_path
        mock_export.assert_called_once_with(voice_state, expected_path)


class TestGenerateCli:
    def test_cli_invocation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 100)

        mock_model = MagicMock()
        mock_model.get_state_for_audio_prompt.return_value = MagicMock()

        with patch("generate.TTSModel.load_model", return_value=mock_model), \
             patch("generate.export_model_state"):
            runner = CliRunner()
            result = runner.invoke(generate, [str(wav_file)])

        assert result.exit_code == 0
        assert "test.safetensors" in result.output
