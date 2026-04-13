# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Wan2.2 S2V TTS integration helpers and examples."""

from __future__ import annotations

import builtins
import importlib.util
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import PIL.Image
import pytest
import soundfile as sf
import torch
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_s2v import Wan22S2VPipeline
from vllm_omni.diffusion.models.wan2_2.wan2_2_s2v_transformer import (
    Wan22S2VBackend,
    _resolve_tts_workdir,
    import_wan_runtime,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_EXAMPLE_PATH = Path(__file__).resolve().parents[4] / 'examples' / 'offline_inference' / 'wan2_2_s2v' / 'end2end.py'


def _load_example_module():
    """Import the offline Wan2.2 S2V example as a reusable test module."""
    module_name = 'tests.diffusion.models.wan2_2._wan22_s2v_end2end'
    spec = importlib.util.spec_from_file_location(module_name, _EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_import_wan_runtime_requires_installed_package(monkeypatch):
    """Verify that Wan imports fail with a clear installation error."""
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "wan" or name.startswith("wan."):
            raise ModuleNotFoundError("No module named 'wan'", name="wan")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "wan", raising=False)
    monkeypatch.delitem(sys.modules, "wan.configs", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="upstream `wan` Python package"):
        import_wan_runtime()


def test_wan22_s2v_forward_exposes_generated_tts_audio_path(tmp_path):
    """Verify that pipeline outputs surface generated TTS audio metadata."""
    prompt_audio = tmp_path / 'tts_prompt.wav'
    prompt_audio.write_bytes(b'RIFF')
    generated_audio = tmp_path / 'generated.wav'
    generated_audio.write_bytes(b'RIFF')

    captured = {}

    pipeline = object.__new__(Wan22S2VPipeline)
    nn.Module.__init__(pipeline)
    pipeline.od_config = OmniDiffusionConfig(model='/tmp/fake-wan22-s2v')
    pipeline.default_fps = 16
    pipeline.enable_diffusion_pipeline_profiler = False

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return torch.zeros((3, 13, 64, 64), dtype=torch.float32), str(generated_audio)

    pipeline.backend = SimpleNamespace(generate=fake_generate)

    request = OmniDiffusionRequest(
        prompts=[
            {
                'prompt': 'A cheerful person is speaking to the camera.',
                'multi_modal_data': {
                    'image': PIL.Image.new('RGB', (64, 64), color=(123, 45, 67)),
                },
            }
        ],
        sampling_params=OmniDiffusionSamplingParams(
            height=64,
            width=64,
            num_frames=17,
            num_inference_steps=2,
            guidance_scale=4.5,
            seed=42,
            extra_args={
                'infer_frames': 16,
                'num_repeat': 1,
                'shift': 3.0,
                'solver_name': 'unipc',
                'enable_tts': True,
                'tts_prompt_audio': str(prompt_audio),
                'tts_prompt_text': 'prompt transcript',
                'tts_text': 'target speech',
            },
        ),
    )

    output = pipeline.forward(request)

    assert output.custom_output['audio_path'] == str(generated_audio)
    assert output.custom_output['fps'] == 16
    assert captured['audio_path'] is None
    assert captured['tts_prompt_audio'] == str(prompt_audio)
    assert captured['tts_prompt_text'] == 'prompt transcript'
    assert captured['tts_text'] == 'target speech'


def test_end2end_merge_audio_uses_generated_tts_audio(monkeypatch, tmp_path):
    """Verify that the offline example merges synthesized audio when present."""
    module = _load_example_module()
    generated_audio = tmp_path / 'generated.wav'
    output_video = tmp_path / 'wan22_s2v.mp4'
    merged = {}

    args = Namespace(
        model='/tmp/fake-model',
        prompt='A cheerful person is speaking to the camera.',
        negative_prompt='',
        image='/tmp/fake-image.png',
        audio=None,
        pose_video=None,
        enable_tts=True,
        tts_prompt_audio='/tmp/fake-prompt.wav',
        tts_prompt_text='prompt transcript',
        tts_text='target speech',
        height=448,
        width=832,
        num_frames=17,
        infer_frames=16,
        num_repeat=1,
        num_inference_steps=2,
        guidance_scale=4.5,
        shift=3.0,
        solver_name='unipc',
        seed=42,
        output=str(output_video),
        enable_cpu_offload=False,
        merge_audio=True,
        enforce_eager=True,
    )

    class FakeOmni:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, prompt, sampling_params):
            return OmniRequestOutput(
                images=[np.zeros((13, 64, 64, 3), dtype=np.float32)],
                _multimodal_output={'fps': 16},
                _custom_output={'audio_path': str(generated_audio)},
            )

    monkeypatch.setattr(module, 'parse_args', lambda: args)
    monkeypatch.setattr(module, 'Omni', FakeOmni)
    monkeypatch.setattr(module, '_save_video', lambda video, output_path, fps: None)
    monkeypatch.setattr(
        module,
        '_merge_audio',
        lambda video_path, audio_path: merged.update(
            {'video_path': video_path, 'audio_path': audio_path}
        ),
    )

    module.main()

    assert merged == {
        'video_path': str(output_video),
        'audio_path': str(generated_audio),
    }


def test_wan22_s2v_load_tts_prompt_audio_accepts_waveform_tensor():
    """Verify that tensor prompt audio bypasses filesystem loading."""
    backend = object.__new__(Wan22S2VBackend)
    waveform = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float32)

    loaded = backend._load_tts_prompt_audio(waveform, target_sr=16000)

    assert torch.equal(loaded, waveform)


def test_wan22_s2v_forward_preserves_zero_seed():
    """Verify that an explicit zero seed is preserved end to end."""
    captured = {}

    pipeline = object.__new__(Wan22S2VPipeline)
    nn.Module.__init__(pipeline)
    pipeline.od_config = OmniDiffusionConfig(model='/tmp/fake-wan22-s2v')
    pipeline.default_fps = 16
    pipeline.enable_diffusion_pipeline_profiler = False

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return torch.zeros((3, 13, 64, 64), dtype=torch.float32), None

    pipeline.backend = SimpleNamespace(generate=fake_generate)

    request = OmniDiffusionRequest(
        prompts=[
            {
                'prompt': 'A cheerful person is speaking to the camera.',
                'multi_modal_data': {
                    'image': PIL.Image.new('RGB', (64, 64), color=(123, 45, 67)),
                    'audio': (np.zeros((16000,), dtype=np.float32), 16000),
                },
            }
        ],
        sampling_params=OmniDiffusionSamplingParams(
            height=64,
            width=64,
            num_frames=17,
            num_inference_steps=2,
            guidance_scale=4.5,
            seed=0,
        ),
    )

    output = pipeline.forward(request)

    assert captured['seed'] == 0
    assert output.custom_output['cleanup_audio_path'] is False


def test_wan22_s2v_forward_accepts_direct_audio_array_argument():
    """Verify that direct audio arrays are normalized into temporary WAV files."""
    captured = {}

    pipeline = object.__new__(Wan22S2VPipeline)
    nn.Module.__init__(pipeline)
    pipeline.od_config = OmniDiffusionConfig(model='/tmp/fake-wan22-s2v')
    pipeline.default_fps = 16
    pipeline.enable_diffusion_pipeline_profiler = False

    def fake_generate(**kwargs):
        captured['audio_path'] = kwargs['audio_path']
        captured['audio_path_exists'] = Path(kwargs['audio_path']).exists()
        return torch.zeros((3, 13, 64, 64), dtype=torch.float32), None

    pipeline.backend = SimpleNamespace(generate=fake_generate)

    request = OmniDiffusionRequest(
        prompts=[
            {
                'prompt': 'A cheerful person is speaking to the camera.',
                'multi_modal_data': {
                    'image': PIL.Image.new('RGB', (64, 64), color=(123, 45, 67)),
                },
            }
        ],
        sampling_params=OmniDiffusionSamplingParams(
            height=64,
            width=64,
            num_frames=17,
            num_inference_steps=2,
            guidance_scale=4.5,
            seed=42,
        ),
    )

    pipeline.forward(request, audio=np.zeros((16000,), dtype=np.float32))

    assert captured['audio_path'].endswith('.wav')
    assert captured['audio_path_exists'] is True


def test_wan22_s2v_forward_does_not_re_normalize_preprocessed_pil_image(monkeypatch):
    """Verify that forward reuses already-normalized PIL images."""
    import vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_s2v as pipeline_module

    captured = {}

    pipeline = object.__new__(Wan22S2VPipeline)
    nn.Module.__init__(pipeline)
    pipeline.od_config = OmniDiffusionConfig(model='/tmp/fake-wan22-s2v')
    pipeline.default_fps = 16
    pipeline.enable_diffusion_pipeline_profiler = False

    def fake_generate(**kwargs):
        captured['image'] = kwargs['image']
        return torch.zeros((3, 13, 64, 64), dtype=torch.float32), None

    pipeline.backend = SimpleNamespace(generate=fake_generate)

    def fail_normalize_image_input(_):
        raise AssertionError('normalize_image_input should not be called for a preprocessed PIL image')

    monkeypatch.setattr(pipeline_module, 'normalize_image_input', fail_normalize_image_input)

    request = OmniDiffusionRequest(
        prompts=[
            {
                'prompt': 'A cheerful person is speaking to the camera.',
                'multi_modal_data': {
                    'image': PIL.Image.new('RGB', (64, 64), color=(123, 45, 67)),
                    'audio': (np.zeros((16000,), dtype=np.float32), 16000),
                },
            }
        ],
        sampling_params=OmniDiffusionSamplingParams(
            height=64,
            width=64,
            num_frames=17,
            num_inference_steps=2,
            guidance_scale=4.5,
            seed=42,
        ),
    )

    pipeline.forward(request)

    assert isinstance(captured['image'], PIL.Image.Image)


def test_end2end_merge_audio_cleans_generated_tts_audio_when_marked_temporary(monkeypatch, tmp_path):
    """Verify that temporary synthesized audio files are removed after muxing."""
    module = _load_example_module()
    generated_audio = tmp_path / 'generated.wav'
    generated_audio.write_bytes(b'RIFF')
    output_video = tmp_path / 'wan22_s2v.mp4'

    args = Namespace(
        model='/tmp/fake-model',
        prompt='A cheerful person is speaking to the camera.',
        negative_prompt='',
        image='/tmp/fake-image.png',
        audio=None,
        pose_video=None,
        enable_tts=True,
        tts_prompt_audio='/tmp/fake-prompt.wav',
        tts_prompt_text='prompt transcript',
        tts_text='target speech',
        height=448,
        width=832,
        num_frames=17,
        infer_frames=16,
        num_repeat=1,
        num_inference_steps=2,
        guidance_scale=4.5,
        shift=3.0,
        solver_name='unipc',
        seed=42,
        output=str(output_video),
        enable_cpu_offload=False,
        merge_audio=True,
        enforce_eager=True,
    )

    class FakeOmni:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, prompt, sampling_params):
            return OmniRequestOutput(
                images=[np.zeros((13, 64, 64, 3), dtype=np.float32)],
                _multimodal_output={'fps': 16},
                _custom_output={
                    'audio_path': str(generated_audio),
                    'cleanup_audio_path': True,
                },
            )

    monkeypatch.setattr(module, 'parse_args', lambda: args)
    monkeypatch.setattr(module, 'Omni', FakeOmni)
    monkeypatch.setattr(module, '_save_video', lambda video, output_path, fps: None)
    monkeypatch.setattr(module, '_merge_audio', lambda video_path, audio_path: None)

    module.main()

    assert not generated_audio.exists()


def test_wan22_s2v_load_tts_prompt_audio_error_mentions_min_sr(tmp_path):
    """Verify that low-sample-rate errors mention the enforced minimum sample rate."""
    prompt_audio = tmp_path / 'low_sr.wav'
    sf.write(prompt_audio, np.zeros((800,), dtype=np.float32), 8000)

    with pytest.raises(ValueError) as exc_info:
        Wan22S2VBackend._load_tts_prompt_audio(str(prompt_audio), target_sr=24000, min_sr=16000)

    message = str(exc_info.value)
    assert '16000' in message
    assert '24000' not in message


def test_resolve_tts_workdir_honors_env_override(monkeypatch, tmp_path):
    """Verify that the TTS workdir environment override takes precedence."""
    override = tmp_path / 'tts-cache'
    monkeypatch.setenv('VLLM_OMNI_WAN22_S2V_TTS_WORKDIR', str(override))

    resolved = _resolve_tts_workdir('/tmp/fake-model')

    assert resolved == override.resolve()


def test_resolve_tts_workdir_falls_back_to_system_temp_when_model_parent_is_not_writable(monkeypatch, tmp_path):
    """Verify that non-writable model parents fall back to the system temp directory."""
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    monkeypatch.delenv('VLLM_OMNI_WAN22_S2V_TTS_WORKDIR', raising=False)

    import vllm_omni.diffusion.models.wan2_2.wan2_2_s2v_transformer as transformer_module

    real_access = transformer_module.os.access

    def fake_access(path, mode):
        if Path(path).resolve() == tmp_path.resolve():
            return False
        return real_access(path, mode)

    monkeypatch.setattr(transformer_module.os, 'access', fake_access)

    resolved = _resolve_tts_workdir(str(model_dir))

    assert resolved == Path(tempfile.gettempdir()).resolve() / 'wan22_s2v_tts_assets'


def test_patch_cosyvoice_audio_io_is_idempotent(monkeypatch):
    """Verify that repeated CosyVoice patch attempts remain safe and deterministic."""
    import types

    cosyvoice_pkg = types.ModuleType('cosyvoice')
    cli_pkg = types.ModuleType('cosyvoice.cli')
    utils_pkg = types.ModuleType('cosyvoice.utils')
    frontend_mod = types.ModuleType('cosyvoice.cli.frontend')
    file_utils_mod = types.ModuleType('cosyvoice.utils.file_utils')
    frontend_mod.load_wav = None
    file_utils_mod.load_wav = None

    monkeypatch.setitem(sys.modules, 'cosyvoice', cosyvoice_pkg)
    monkeypatch.setitem(sys.modules, 'cosyvoice.cli', cli_pkg)
    monkeypatch.setitem(sys.modules, 'cosyvoice.utils', utils_pkg)
    monkeypatch.setitem(sys.modules, 'cosyvoice.cli.frontend', frontend_mod)
    monkeypatch.setitem(sys.modules, 'cosyvoice.utils.file_utils', file_utils_mod)

    backend = object.__new__(Wan22S2VBackend)
    backend._patch_cosyvoice_audio_io()
    first_frontend = frontend_mod.load_wav
    first_file_utils = file_utils_mod.load_wav

    frontend_mod.load_wav = object()
    file_utils_mod.load_wav = object()
    backend._patch_cosyvoice_audio_io()

    assert getattr(frontend_mod, '_vllm_omni_load_wav_patch_applied', False) is True
    assert getattr(file_utils_mod, '_vllm_omni_load_wav_patch_applied', False) is True
    assert frontend_mod.load_wav is not first_frontend
    assert file_utils_mod.load_wav is not first_file_utils


def test_end2end_merge_audio_reports_removed_temporary_tts_audio(monkeypatch, tmp_path, capsys):
    """Verify that the offline example reports when a temporary audio file is removed."""
    module = _load_example_module()
    generated_audio = tmp_path / 'generated.wav'
    generated_audio.write_bytes(b'RIFF')
    output_video = tmp_path / 'wan22_s2v.mp4'

    args = Namespace(
        model='/tmp/fake-model',
        prompt='A cheerful person is speaking to the camera.',
        negative_prompt='',
        image='/tmp/fake-image.png',
        audio=None,
        pose_video=None,
        enable_tts=True,
        tts_prompt_audio='/tmp/fake-prompt.wav',
        tts_prompt_text='prompt transcript',
        tts_text='target speech',
        height=448,
        width=832,
        num_frames=17,
        infer_frames=16,
        num_repeat=1,
        num_inference_steps=2,
        guidance_scale=4.5,
        shift=3.0,
        solver_name='unipc',
        seed=42,
        output=str(output_video),
        enable_cpu_offload=False,
        merge_audio=True,
        enforce_eager=True,
    )

    class FakeOmni:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, prompt, sampling_params):
            return OmniRequestOutput(
                images=[np.zeros((13, 64, 64, 3), dtype=np.float32)],
                _multimodal_output={'fps': 16},
                _custom_output={
                    'audio_path': str(generated_audio),
                    'cleanup_audio_path': True,
                },
            )

    monkeypatch.setattr(module, 'parse_args', lambda: args)
    monkeypatch.setattr(module, 'Omni', FakeOmni)
    monkeypatch.setattr(module, '_save_video', lambda video, output_path, fps: None)
    monkeypatch.setattr(module, '_merge_audio', lambda video_path, audio_path: None)

    module.main()
    stdout = capsys.readouterr().out

    assert 'temporary file was removed' in stdout
    assert str(generated_audio) not in stdout
