from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from speak import get_voice, speak, generate_wav, convert_to_ogg, _repair_wav_header, _generate_waveform

SAMPLE_CONFIG = """\
[agents.main]
voice = "eponine"

[agents.protocol]
voice = "gabor"

[agents.novoice]
channel = "1474167721808756746"
"""


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(SAMPLE_CONFIG)
    return path


@pytest.fixture
def tensors_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create an empty tensors/ directory and monkeypatch speak.TENSORS_DIR."""
    directory = tmp_path / "tensors"
    directory.mkdir()
    monkeypatch.setattr("speak.TENSORS_DIR", directory)
    return directory


class TestGetVoice:
    def test_valid_agent(self, config_path: Path) -> None:
        voice = get_voice("main", config_path=config_path)
        assert voice == "eponine"

    def test_unknown_agent(self, config_path: Path, tensors_dir: Path) -> None:
        voice = get_voice("nonexistent", config_path=config_path)
        assert voice == "eponine"

    def test_second_agent(self, config_path: Path, tensors_dir: Path) -> None:
        (tensors_dir / "gabor.safetensors").write_bytes(b"fake tensor data")

        voice = get_voice("protocol", config_path=config_path)
        assert voice == str(tensors_dir / "gabor.safetensors")


class TestGetVoiceResolution:
    """Tests for extended voice resolution logic in get_voice."""

    def test_agent_without_voice_key_falls_back_to_eponine(
        self, config_path: Path, tensors_dir: Path
    ) -> None:
        """When an agent has no 'voice' key, get_voice should return 'eponine'."""
        voice = get_voice("novoice", config_path=config_path)
        assert voice == "eponine"

    def test_voice_matching_safetensors_file_returns_full_path(
        self, tmp_path: Path, tensors_dir: Path
    ) -> None:
        """When a .safetensors file matching the voice name exists in tensors/,
        get_voice should return the full path to that file."""
        (tensors_dir / "gabor-mate.safetensors").write_bytes(b"fake tensor data")

        config_with_custom_voice = """\
[agents.protocol]
voice = "gabor-mate"
"""
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text(config_with_custom_voice)

        voice = get_voice("protocol", config_path=cfg_path)
        assert voice == str(tensors_dir / "gabor-mate.safetensors")

    def test_builtin_voice_name_passes_through_as_is(
        self, tmp_path: Path, tensors_dir: Path
    ) -> None:
        """When the voice name is a built-in pocket-tts voice, get_voice
        should return it as-is without modification."""
        config_with_builtin = """\
[agents.reader]
voice = "alba"
"""
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text(config_with_builtin)

        voice = get_voice("reader", config_path=cfg_path)
        assert voice == "alba"

    def test_unknown_voice_raises_click_exception(
        self, config_path: Path, tensors_dir: Path
    ) -> None:
        """When the voice is not a safetensors file and not a built-in voice,
        get_voice should raise click.ClickException."""
        # 'protocol' agent has voice = "gabor" which is neither built-in nor a safetensors file
        with pytest.raises(click.ClickException, match="Unknown voice"):
            get_voice("protocol", config_path=config_path)


class TestSpeakCli:
    @patch("speak.send_voice_message")
    @patch("speak._generate_waveform", return_value=(b"\x00" * 256, 1.0))
    @patch("speak.convert_to_ogg")
    @patch("speak.generate_wav")
    def test_valid_invocation(
        self,
        mock_generate_wav: MagicMock,
        mock_convert_to_ogg: MagicMock,
        mock_waveform: MagicMock,
        mock_send: MagicMock,
        config_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr("speak.CONFIG_PATH", config_path)
        mock_generate_wav.return_value = tmp_path / "output.wav"
        mock_convert_to_ogg.return_value = tmp_path / "output.ogg"

        runner = CliRunner()
        result = runner.invoke(
            speak,
            ["main", "1474167721808756746"],
            input="my-token\nHello from CLI\n",
        )
        assert result.exit_code == 0
        assert "Agent: main" in result.output
        assert "Text: Hello from CLI" in result.output
        assert "my-token" not in result.output

    @patch("speak.send_voice_message")
    @patch("speak._generate_waveform", return_value=(b"\x00" * 256, 1.0))
    @patch("speak.convert_to_ogg")
    @patch("speak.generate_wav")
    def test_unknown_agent_cli(
        self,
        mock_generate_wav: MagicMock,
        mock_convert_to_ogg: MagicMock,
        mock_waveform: MagicMock,
        mock_send: MagicMock,
        config_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        tensors_dir: Path,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr("speak.CONFIG_PATH", config_path)
        mock_generate_wav.return_value = tmp_path / "output.wav"
        mock_convert_to_ogg.return_value = tmp_path / "output.ogg"
        runner = CliRunner()
        result = runner.invoke(
            speak,
            ["nonexistent", "1474167721808756746"],
            input="token\ntext\n",
        )
        assert result.exit_code == 0
        assert "Voice: eponine" in result.output


class TestGenerateWav:
    def test_accepts_string_output_dir(
        self, tmp_path: Path
    ) -> None:
        with patch("speak.TTSModel") as MockTTSModel, \
             patch("scipy.io.wavfile.write") as mock_wavfile_write, \
             patch("speak._repair_wav_header"):
            mock_model = MagicMock()
            mock_model.generate_audio.return_value = MagicMock(numpy=MagicMock())
            mock_model.sample_rate = 24000
            MockTTSModel.load_model.return_value = mock_model

            result_path = generate_wav("test", "alba", str(tmp_path))

            assert result_path == tmp_path / "output.wav"

    def test_generate_wav_loads_model_gets_voice_state_and_saves_wav(
        self, tmp_path: Path
    ) -> None:
        with patch("speak.TTSModel") as MockTTSModel, \
             patch("scipy.io.wavfile.write") as mock_wavfile_write, \
             patch("speak._repair_wav_header"):
            # Arrange
            fake_voice_state = {"some": "state", "data": [1, 2, 3]}
            fake_audio = MagicMock()
            fake_numpy = MagicMock()
            fake_audio.numpy.return_value = fake_numpy
            fake_sample_rate = 24000

            mock_model = MagicMock()
            mock_model.get_state_for_audio_prompt.return_value = fake_voice_state
            mock_model.generate_audio.return_value = fake_audio
            mock_model.sample_rate = fake_sample_rate
            MockTTSModel.load_model.return_value = mock_model

            text = "Hello, world"
            voice = "eponine"
            output_dir = tmp_path
            expected_output_path = output_dir / "output.wav"

            # Act
            result_path = generate_wav(text, voice, output_dir)

            # Assert
            MockTTSModel.load_model.assert_called_once()
            mock_model.get_state_for_audio_prompt.assert_called_once_with(voice)
            mock_model.generate_audio.assert_called_once_with(fake_voice_state, text)
            mock_wavfile_write.assert_called_once_with(
                str(expected_output_path), fake_sample_rate, fake_numpy
            )
            assert result_path == expected_output_path


class TestConvertToOgg:
    def test_converts_wav_to_ogg_with_default_bitrate(
        self, tmp_path: Path
    ) -> None:
        wav_path = tmp_path / "output.wav"
        wav_path.write_bytes(b"RIFF" + b"\x00" * 100)

        mock_stream = MagicMock()
        mock_output = MagicMock()
        mock_stream.output.return_value = mock_output

        with patch("ffmpeg.input", return_value=mock_stream) as mock_input:
            result_path = convert_to_ogg(str(wav_path))

            mock_input.assert_called_once_with(str(wav_path))
            mock_stream.output.assert_called_once()
            call_args = mock_stream.output.call_args
            assert call_args[0][0].endswith(".ogg")
            mock_output.run.assert_called_once_with(
                overwrite_output=True, capture_stdout=True, capture_stderr=True
            )
            assert result_path == Path(str(wav_path).replace(".wav", ".ogg"))

    def test_uses_custom_bitrate(
        self, tmp_path: Path
    ) -> None:
        wav_path = tmp_path / "output.wav"
        wav_path.write_bytes(b"RIFF" + b"\x00" * 100)

        mock_stream = MagicMock()
        mock_output = MagicMock()
        mock_stream.output.return_value = mock_output

        with patch("ffmpeg.input", return_value=mock_stream):
            convert_to_ogg(str(wav_path), bitrate="48k")

            call_args = mock_stream.output.call_args
            kwargs = call_args[1]
            assert kwargs["audio_bitrate"] == "48k"

    def test_defaults_to_32k_bitrate(
        self, tmp_path: Path
    ) -> None:
        wav_path = tmp_path / "output.wav"
        wav_path.write_bytes(b"RIFF" + b"\x00" * 100)

        mock_stream = MagicMock()
        mock_output = MagicMock()
        mock_stream.output.return_value = mock_output

        with patch("ffmpeg.input", return_value=mock_stream):
            convert_to_ogg(str(wav_path))

            call_args = mock_stream.output.call_args
            kwargs = call_args[1]
            assert kwargs["audio_bitrate"] == "32k"


class TestRepairWavHeader:
    def test_fixes_riff_and_data_chunk_sizes(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "test.wav"
        # Write a minimal valid-ish WAV with wrong chunk sizes
        with open(wav_path, "wb") as f:
            f.write(b"RIFF")
            f.write(b"\xff\xff\xff\xff")  # wrong RIFF size (offset 4)
            f.write(b"WAVEfmt ")
            f.write(b"\x10\x00\x00\x00")  # chunk size 16
            f.write(b"\x01\x00\x01\x00")  # PCM, mono
            f.write(b"\x40\x1f\x00\x00")  # 8000 Hz sample rate
            f.write(b"\x80\x3e\x00\x00")  # byte rate
            f.write(b"\x02\x00\x08\x00")  # block align, bits per sample
            f.write(b"data")
            f.write(b"\xff\xff\xff\xff")  # wrong data size (offset 40)
            f.write(b"\x00\x00\x00\x00")  # 1 sample

        _repair_wav_header(wav_path)

        with open(wav_path, "rb") as f:
            f.seek(4)
            riff_size = int.from_bytes(f.read(4), "little")
            f.seek(40)
            data_size = int.from_bytes(f.read(4), "little")

        file_size = wav_path.stat().st_size
        assert riff_size == file_size - 8
        assert data_size == file_size - 44


class TestGenerateWaveform:
    def test_returns_256_bytes(self, tmp_path: Path) -> None:
        import struct
        import wave

        wav_path = tmp_path / "test.wav"
        sample_rate = 24000
        samples = b"\x00\x10\x20\x30\x40\x50\x60\x70\x80\x90"
        with wave.open(str(wav_path), "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframesraw(samples)

        waveform, duration = _generate_waveform(wav_path)
        assert len(waveform) == 256
        assert duration > 0

    def test_empty_audio_returns_flat_waveform(self, tmp_path: Path) -> None:
        import wave

        wav_path = tmp_path / "empty.wav"
        with wave.open(str(wav_path), "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)

        waveform, duration = _generate_waveform(wav_path)
        assert waveform == b"\x00" * 256
        assert duration == 0.0
