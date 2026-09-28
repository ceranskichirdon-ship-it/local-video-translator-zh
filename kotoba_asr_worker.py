from __future__ import annotations
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid


def emit(kind, message, progress=None):
    row = {'type': kind, 'stage': 'transcribe', 'message': message}
    if progress is not None:
        row['progress'] = progress
    print(json.dumps(row, ensure_ascii=False), flush=True)


def prepare_offline(root: Path):
    cache = root / 'work' / 'kotoba-cache'
    envs = {'HF_HOME': cache / 'huggingface', 'HF_HUB_CACHE': cache / 'huggingface' / 'hub',
            'TORCH_HOME': cache / 'torch', 'TMP': root / 'work' / 'kotoba-temp',
            'TEMP': root / 'work' / 'kotoba-temp', 'NUMBA_CACHE_DIR': root / 'work' / 'numba',
            'CUDA_CACHE_PATH': root / 'work' / 'cuda-cache'}
    for key, value in envs.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1', TOKENIZERS_PARALLELISM='false')


def publish(path: Path, value: dict):
    if path.exists():
        raise RuntimeError(f'识别输出已存在，为保护已有文件已停止：{path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('kotoba_result_' + uuid.uuid4().hex + '.json.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args):
    root = Path(__file__).resolve().parent
    audio_path = Path(args.audio).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not audio_path.is_file():
        raise RuntimeError(f'找不到音频：{audio_path}')
    if output_path == audio_path or output_path.exists():
        raise RuntimeError('识别结果不能覆盖音频或已有文件。')
    for required in ('config.json', 'model.safetensors', 'kotoba_whisper.py'):
        if not (model_path / required).is_file():
            raise RuntimeError(f'Kotoba 模型文件不完整：{model_path / required}')
    prepare_offline(root)
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import pipeline

    waveform, sample_rate = sf.read(str(audio_path), dtype='float32', always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or waveform.size == 0:
        raise RuntimeError('音频不是有效的单声道采样数据。')
    waveform = np.asarray(waveform, dtype=np.float32)
    if sample_rate != 16000:
        import torchaudio.functional as audio_functional
        waveform = audio_functional.resample(torch.from_numpy(waveform), sample_rate, 16000).numpy()
        sample_rate = 16000
    duration = len(waveform) / sample_rate
    if duration <= 0:
        raise RuntimeError('音频时长无效。')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device.startswith('cuda') else torch.float32
    device_name = torch.cuda.get_device_name(0) if device.startswith('cuda') else 'CPU'
    emit('log', f'加载本地 Kotoba-Whisper v2.1 日语模型：{device_name}；音频 {duration:.1f} 秒')
    emit('progress', '正在加载日语专用识别模型', 3)
    started = time.monotonic()
    # Kotoba v2.1's custom stable-ts postprocessor indexes the first element
    # of a 1-D audio array; for chunked long audio this turns it into a scalar
    # and crashes while reading shape[-1]. The base Whisper pipeline already
    # returns chunk timestamps, so keep those and skip the faulty refinement.
    recognizer = pipeline(
        model=str(model_path), device=device, torch_dtype=dtype,
        model_kwargs={'attn_implementation': 'sdpa', 'local_files_only': True},
        trust_remote_code=True, punctuator=False, stable_ts=False, batch_size=1)
    emit('progress', '正在识别日语语音并生成模型时间戳', 16)
    # Let Whisper's own long-form decoder split the recording internally.
    # Transformers' external chunk_length_s path can truncate timestamp tokens
    # at chunk edges, so several returned chunks have a missing end timestamp.
    with torch.inference_mode():
        result = recognizer({'array': waveform, 'sampling_rate': sample_rate},
                            return_timestamps=True, generate_kwargs={'language': 'ja', 'task': 'transcribe'})
    raw_segments = result.get('chunks') or []
    segments = []
    for chunk in raw_segments:
        text = str(chunk.get('text') or '').strip()
        stamp = chunk.get('timestamp')
        if not text:
            continue
        if not isinstance(stamp, (tuple, list)) or len(stamp) != 2 or stamp[0] is None or stamp[1] is None:
            raise RuntimeError('Kotoba 返回了没有有效时间戳的文字，已停止以免丢失字幕。')
        start, end = float(stamp[0]), float(stamp[1])
        if not (0 <= start < end <= duration + 0.5):
            raise RuntimeError('Kotoba 返回的字幕时间超出音频范围。')
        segments.append({'id': len(segments) + 1, 'start': round(start, 3),
                         'end': round(min(end, duration), 3), 'text': text})
    if not segments:
        raise RuntimeError('Kotoba 没有识别到有效日语。')
    full_text = str(result.get('text') or ''.join(s['text'] for s in segments))
    payload = {'language': 'ja', 'language_name': 'Japanese', 'text': full_text,
               'segments': segments, 'backend': 'kotoba-whisper-v2.1',
               'alignment': 'kotoba-whisper-native-longform-timestamps', 'model': str(model_path), 'audio': str(audio_path),
               'duration': duration, 'device': device_name, 'dtype': str(dtype),
               'elapsed_seconds': round(time.monotonic() - started, 3), 'schema_version': 1}
    publish(output_path, payload)
    emit('progress', f'日语识别完成：{len(segments)} 条字幕，Kotoba-Whisper v2.1', 100)


def main(argv=None):
    parser = argparse.ArgumentParser(description='本地 Kotoba 日语语音识别与时间戳对齐')
    parser.add_argument('--audio', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            run(args)
        return 0
    except KeyboardInterrupt:
        emit('error', '日语识别已取消。')
        return 130
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        message = str(exc)
        if 'out of memory' in message.lower():
            message = '显存不足，请关闭其他使用显卡的程序后重试。' + message
        emit('error', message or type(exc).__name__)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
